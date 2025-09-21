-- Database Schema for Phase 9 Autonomous Development Teams
-- This schema supports team assembly, workflow orchestration, performance tracking,
-- and cross-project knowledge synthesis for autonomous AI development teams.

-- ============================================================================
-- AUTONOMOUS TEAMS CORE TABLES
-- ============================================================================

-- Autonomous development teams
CREATE TABLE IF NOT EXISTS autonomous_teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    project_name TEXT NOT NULL,
    project_description TEXT,
    composition JSONB NOT NULL, -- Full team composition data
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'completed', 'cancelled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Team metrics
    total_agents INTEGER DEFAULT 0,
    total_cost DECIMAL(10,2) DEFAULT 0.00,
    estimated_efficiency DECIMAL(3,2) DEFAULT 0.00,
    
    -- Indexing
    CONSTRAINT autonomous_teams_name_key UNIQUE (name)
);

-- Team agents and their capabilities
CREATE TABLE IF NOT EXISTS team_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES autonomous_teams(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    primary_role TEXT NOT NULL,
    skills JSONB DEFAULT '{}', -- Skill name -> proficiency level
    cost_per_hour DECIMAL(8,2) DEFAULT 0.00,
    availability DECIMAL(3,2) DEFAULT 1.00, -- 0.0 to 1.0
    specialization_score DECIMAL(3,2) DEFAULT 0.00,
    experience_level INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE (team_id, agent_id)
);

-- ============================================================================
-- WORKFLOW ORCHESTRATION TABLES
-- ============================================================================

-- Workflow executions
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL,
    team_id UUID REFERENCES autonomous_teams(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    description TEXT,
    template_name TEXT NOT NULL DEFAULT 'web_application',
    
    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    current_phase TEXT,
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_duration_hours DECIMAL(8,2) DEFAULT 0.00,
    actual_duration_hours DECIMAL(8,2) DEFAULT 0.00,
    
    -- Success metrics
    success_rate DECIMAL(3,2) DEFAULT 0.00,
    total_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    
    -- Configuration
    team_composition JSONB DEFAULT '{}',
    configuration JSONB DEFAULT '{}'
);

-- Workflow tasks
CREATE TABLE IF NOT EXISTS workflow_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    phase TEXT NOT NULL,
    assigned_agent TEXT,
    
    -- Requirements
    required_skills TEXT[] DEFAULT '{}',
    dependencies UUID[] DEFAULT '{}', -- Array of task IDs this depends on
    
    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'ready', 'in_progress', 'blocked', 'review_required', 'completed', 'failed', 'cancelled')),
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_hours DECIMAL(6,2) DEFAULT 0.00,
    actual_hours DECIMAL(6,2) DEFAULT 0.00,
    
    -- Quality and deliverables
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    quality_gates TEXT[] DEFAULT '{}',
    deliverables TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- ============================================================================
-- PERFORMANCE TRACKING TABLES
-- ============================================================================

-- Team performance profiles
CREATE TABLE IF NOT EXISTS team_performance_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES autonomous_teams(id) ON DELETE CASCADE,
    team_name TEXT NOT NULL,
    maturity_level TEXT NOT NULL DEFAULT 'forming' CHECK (maturity_level IN ('forming', 'storming', 'norming', 'performing', 'transforming')),
    
    -- Core metrics (current values)
    task_completion_rate DECIMAL(3,2) DEFAULT 0.00,
    average_task_duration DECIMAL(8,2) DEFAULT 0.00,
    quality_score DECIMAL(3,2) DEFAULT 0.00,
    bug_introduction_rate DECIMAL(3,2) DEFAULT 0.00,
    collaboration_efficiency DECIMAL(3,2) DEFAULT 0.00,
    resource_utilization DECIMAL(3,2) DEFAULT 0.00,
    client_satisfaction DECIMAL(3,2) DEFAULT 0.00,
    technical_debt_accumulation DECIMAL(3,2) DEFAULT 0.00,
    innovation_index DECIMAL(3,2) DEFAULT 0.00,
    knowledge_sharing_score DECIMAL(3,2) DEFAULT 0.00,
    
    -- Performance analytics
    performance_score DECIMAL(3,2) DEFAULT 0.00, -- Overall score 0-10
    consistency_score DECIMAL(3,2) DEFAULT 0.00, -- Consistency score 0-10
    improvement_rate DECIMAL(5,4) DEFAULT 0.00,  -- Rate of improvement
    
    -- Team characteristics
    team_size INTEGER DEFAULT 0,
    skill_distribution JSONB DEFAULT '{}',
    specialization_balance DECIMAL(3,2) DEFAULT 0.00,
    
    -- Collaboration metrics
    communication_frequency DECIMAL(5,2) DEFAULT 0.00,
    knowledge_sharing_events INTEGER DEFAULT 0,
    cross_training_sessions INTEGER DEFAULT 0,
    mentoring_relationships INTEGER DEFAULT 0,
    
    -- Analysis results
    strengths TEXT[] DEFAULT '{}',
    bottlenecks TEXT[] DEFAULT '{}',
    risk_factors TEXT[] DEFAULT '{}',
    optimization_opportunities TEXT[] DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (team_id)
);

-- Performance data points (historical tracking)
CREATE TABLE IF NOT EXISTS performance_data_points (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES autonomous_teams(id) ON DELETE CASCADE,
    metric TEXT NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Context
    project_id TEXT,
    agent_id TEXT,
    context JSONB DEFAULT '{}',
    
    -- Indexing for time-series queries
    INDEX idx_performance_data_team_metric_time (team_id, metric, timestamp DESC),
    INDEX idx_performance_data_timestamp (timestamp DESC)
);

-- Performance optimizations and recommendations
CREATE TABLE IF NOT EXISTS performance_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES autonomous_teams(id) ON DELETE CASCADE,
    optimization_type TEXT NOT NULL,
    description TEXT NOT NULL,
    
    -- Impact and effort
    expected_impact DECIMAL(3,2) DEFAULT 0.00, -- 0.0 to 1.0
    implementation_effort TEXT NOT NULL DEFAULT 'medium' CHECK (implementation_effort IN ('low', 'medium', 'high')),
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    success_probability DECIMAL(3,2) DEFAULT 0.00,
    time_to_impact_days INTEGER DEFAULT 7,
    
    -- Implementation details
    prerequisites TEXT[] DEFAULT '{}',
    implementation_steps TEXT[] DEFAULT '{}',
    success_metrics TEXT[] DEFAULT '{}',
    
    -- Status tracking
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    implemented_at TIMESTAMPTZ,
    
    INDEX idx_optimization_team_priority (team_id, priority DESC)
);

-- ============================================================================
-- CROSS-PROJECT KNOWLEDGE TABLES
-- ============================================================================

-- Project metrics for pattern analysis
CREATE TABLE IF NOT EXISTS cross_project_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'partial_success', 'failure', 'cancelled')),
    
    -- Project characteristics
    completion_time_hours DECIMAL(8,2) DEFAULT 0.00,
    estimated_time_hours DECIMAL(8,2) DEFAULT 0.00,
    team_size INTEGER DEFAULT 0,
    technologies_used TEXT[] DEFAULT '{}',
    architectural_patterns TEXT[] DEFAULT '{}',
    
    -- Quality metrics
    test_coverage DECIMAL(3,2) DEFAULT 0.00,
    performance_score DECIMAL(3,2) DEFAULT 0.00,
    security_score DECIMAL(3,2) DEFAULT 0.00,
    code_quality_score DECIMAL(3,2) DEFAULT 0.00,
    bug_count INTEGER DEFAULT 0,
    critical_bugs INTEGER DEFAULT 0,
    
    -- Success indicators
    deployment_success BOOLEAN DEFAULT TRUE,
    user_satisfaction DECIMAL(3,2) DEFAULT 0.00,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_cross_project_outcome (outcome),
    INDEX idx_cross_project_technologies (technologies_used),
    INDEX idx_cross_project_team_size (team_size)
);

-- Identified patterns from cross-project analysis
CREATE TABLE IF NOT EXISTS identified_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type TEXT NOT NULL CHECK (pattern_type IN ('success_pattern', 'anti_pattern', 'optimization_pattern', 'architectural_pattern', 'testing_pattern', 'deployment_pattern', 'performance_pattern', 'security_pattern')),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    context TEXT,
    
    -- Evidence and confidence
    evidence_projects TEXT[] DEFAULT '{}',
    confidence TEXT NOT NULL DEFAULT 'low' CHECK (confidence IN ('low', 'medium', 'high', 'very_high')),
    success_correlation DECIMAL(3,2) DEFAULT 0.00, -- -1.0 to 1.0
    frequency INTEGER DEFAULT 0,
    impact_score DECIMAL(5,2) DEFAULT 0.00,
    
    -- Conditions and guidance
    conditions TEXT[] DEFAULT '{}',
    implementation_guide TEXT,
    related_patterns UUID[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    
    -- Discovery metadata
    discovered_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_patterns_type_confidence (pattern_type, confidence),
    INDEX idx_patterns_tags (tags),
    INDEX idx_patterns_correlation (success_correlation DESC)
);

-- Knowledge syntheses (comprehensive analysis reports)
CREATE TABLE IF NOT EXISTS knowledge_syntheses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    patterns_analyzed INTEGER DEFAULT 0,
    projects_analyzed INTEGER DEFAULT 0,
    
    -- Key findings
    key_insights TEXT[] DEFAULT '{}',
    recommendations TEXT[] DEFAULT '{}',
    risk_factors TEXT[] DEFAULT '{}',
    success_predictors TEXT[] DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_synthesis_created (created_at DESC)
);

-- ============================================================================
-- GLOBAL KNOWLEDGE NETWORK TABLES
-- ============================================================================

-- Knowledge items in the global network
CREATE TABLE IF NOT EXISTS global_knowledge_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_type TEXT NOT NULL CHECK (knowledge_type IN ('pattern', 'anti_pattern', 'best_practice', 'solution_template', 'performance_benchmark', 'tool_recommendation', 'architecture_pattern', 'process_improvement')),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    content JSONB NOT NULL,
    
    -- Classification
    domain TEXT NOT NULL,
    technologies TEXT[] DEFAULT '{}',
    complexity_level INTEGER DEFAULT 5 CHECK (complexity_level >= 1 AND complexity_level <= 10),
    
    -- Quality metrics
    success_rate DECIMAL(3,2) DEFAULT 0.00,
    confidence_score DECIMAL(3,2) DEFAULT 0.00,
    validation_score DECIMAL(3,2) DEFAULT 0.00,
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    success_feedback INTEGER DEFAULT 0,
    failure_feedback INTEGER DEFAULT 0,
    
    -- Privacy and attribution
    privacy_level TEXT NOT NULL DEFAULT 'anonymized' CHECK (privacy_level IN ('public', 'anonymized', 'aggregated', 'encrypted', 'private')),
    source_organization TEXT,
    contributor_id TEXT,
    anonymized_source TEXT,
    
    -- Network metadata
    last_validated TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    -- Relationships
    related_items UUID[] DEFAULT '{}',
    dependencies UUID[] DEFAULT '{}',
    supersedes UUID[] DEFAULT '{}',
    
    INDEX idx_knowledge_type_domain (knowledge_type, domain),
    INDEX idx_knowledge_technologies (technologies),
    INDEX idx_knowledge_success_rate (success_rate DESC),
    INDEX idx_knowledge_validation (validation_score DESC),
    INDEX idx_knowledge_usage (usage_count DESC)
);

-- Network nodes (organizations/teams in the network)
CREATE TABLE IF NOT EXISTS network_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    organization_type TEXT NOT NULL DEFAULT 'development_team',
    role TEXT NOT NULL DEFAULT 'contributor' CHECK (role IN ('contributor', 'consumer', 'validator', 'coordinator', 'researcher')),
    
    -- Capabilities
    domains TEXT[] DEFAULT '{}',
    technologies TEXT[] DEFAULT '{}',
    expertise_level JSONB DEFAULT '{}',
    
    -- Network participation
    knowledge_contributed INTEGER DEFAULT 0,
    knowledge_consumed INTEGER DEFAULT 0,
    validation_activity INTEGER DEFAULT 0,
    reputation_score DECIMAL(3,2) DEFAULT 5.00,
    trust_level DECIMAL(3,2) DEFAULT 5.00,
    
    -- Collaboration settings
    collaboration_openness DECIMAL(3,2) DEFAULT 0.50,
    preferred_privacy_level TEXT DEFAULT 'anonymized',
    data_sharing_policy JSONB DEFAULT '{}',
    
    -- Connection details
    api_endpoint TEXT,
    webhook_url TEXT,
    encryption_key TEXT,
    
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_network_nodes_role (role),
    INDEX idx_network_nodes_active (last_active DESC),
    INDEX idx_network_nodes_reputation (reputation_score DESC)
);

-- Knowledge network queries (for analytics and caching)
CREATE TABLE IF NOT EXISTS network_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id TEXT NOT NULL UNIQUE,
    requester_id TEXT NOT NULL,
    
    -- Query parameters
    domain TEXT,
    technologies TEXT[] DEFAULT '{}',
    knowledge_types TEXT[] DEFAULT '{}',
    complexity_range INTEGER[] DEFAULT '{1,10}',
    min_success_rate DECIMAL(3,2) DEFAULT 0.00,
    min_confidence DECIMAL(3,2) DEFAULT 0.00,
    
    -- Result preferences
    max_results INTEGER DEFAULT 50,
    preferred_privacy_level TEXT DEFAULT 'anonymized',
    include_experimental BOOLEAN DEFAULT FALSE,
    
    -- Results tracking
    results_count INTEGER DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_queries_requester (requester_id),
    INDEX idx_queries_created (created_at DESC)
);

-- ============================================================================
-- ANALYTICS AND REPORTING VIEWS
-- ============================================================================

-- Team performance summary view
CREATE OR REPLACE VIEW team_performance_summary AS
SELECT 
    tp.team_id,
    at.name as team_name,
    tp.performance_score,
    tp.consistency_score,
    tp.maturity_level,
    tp.team_size,
    COUNT(pdp.id) as total_data_points,
    AVG(pdp.value) FILTER (WHERE pdp.metric = 'task_completion_rate') as avg_completion_rate,
    AVG(pdp.value) FILTER (WHERE pdp.metric = 'quality_score') as avg_quality_score,
    tp.last_updated
FROM team_performance_profiles tp
JOIN autonomous_teams at ON tp.team_id = at.id
LEFT JOIN performance_data_points pdp ON tp.team_id = pdp.team_id
    AND pdp.timestamp > NOW() - INTERVAL '30 days'
GROUP BY tp.team_id, at.name, tp.performance_score, tp.consistency_score, 
         tp.maturity_level, tp.team_size, tp.last_updated;

-- Project success analysis view
CREATE OR REPLACE VIEW project_success_analysis AS
SELECT 
    outcome,
    COUNT(*) as project_count,
    AVG(completion_time_hours) as avg_completion_time,
    AVG(team_size) as avg_team_size,
    AVG(test_coverage) as avg_test_coverage,
    AVG(performance_score) as avg_performance,
    AVG(user_satisfaction) as avg_satisfaction,
    ARRAY_AGG(DISTINCT unnest(technologies_used)) as common_technologies
FROM cross_project_metrics
GROUP BY outcome;

-- Knowledge network statistics view
CREATE OR REPLACE VIEW knowledge_network_stats AS
SELECT 
    COUNT(*) as total_knowledge_items,
    COUNT(DISTINCT domain) as unique_domains,
    COUNT(DISTINCT unnest(technologies)) as unique_technologies,
    AVG(success_rate) as avg_success_rate,
    AVG(validation_score) as avg_validation_score,
    COUNT(*) FILTER (WHERE usage_count > 0) as used_items,
    COUNT(*) FILTER (WHERE validation_score >= 8.0) as high_quality_items,
    MAX(created_at) as latest_contribution
FROM global_knowledge_items;

-- Workflow efficiency analysis view
CREATE OR REPLACE VIEW workflow_efficiency_analysis AS
SELECT 
    template_name,
    status,
    COUNT(*) as workflow_count,
    AVG(actual_duration_hours) as avg_duration,
    AVG(success_rate) as avg_success_rate,
    AVG(completed_tasks::decimal / NULLIF(total_tasks, 0)) as avg_completion_rate,
    AVG(actual_duration_hours / NULLIF(estimated_duration_hours, 0)) as time_estimation_accuracy
FROM workflow_executions
WHERE started_at IS NOT NULL
GROUP BY template_name, status;

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Time-series performance data indexes
CREATE INDEX IF NOT EXISTS idx_performance_data_recent 
ON performance_data_points (team_id, timestamp DESC) 
WHERE timestamp > NOW() - INTERVAL '90 days';

-- Workflow task dependency analysis
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_dependencies 
ON workflow_tasks USING GIN (dependencies);

-- Knowledge search optimization
CREATE INDEX IF NOT EXISTS idx_knowledge_search 
ON global_knowledge_items USING GIN (to_tsvector('english', title || ' ' || description));

-- Pattern analysis optimization
CREATE INDEX IF NOT EXISTS idx_patterns_evidence 
ON identified_patterns USING GIN (evidence_projects);

-- ============================================================================
-- TRIGGERS FOR DATA CONSISTENCY
-- ============================================================================

-- Update team performance profile when new data is added
CREATE OR REPLACE FUNCTION update_team_performance_profile()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE team_performance_profiles
    SET last_updated = NEW.timestamp
    WHERE team_id = NEW.team_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_performance_profile
    AFTER INSERT ON performance_data_points
    FOR EACH ROW
    EXECUTE FUNCTION update_team_performance_profile();

-- Update workflow execution statistics
CREATE OR REPLACE FUNCTION update_workflow_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE workflow_executions
    SET 
        completed_tasks = (
            SELECT COUNT(*) FROM workflow_tasks 
            WHERE workflow_id = NEW.workflow_id AND status = 'completed'
        ),
        failed_tasks = (
            SELECT COUNT(*) FROM workflow_tasks 
            WHERE workflow_id = NEW.workflow_id AND status = 'failed'
        )
    WHERE id = NEW.workflow_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_workflow_stats
    AFTER UPDATE OF status ON workflow_tasks
    FOR EACH ROW
    WHEN (OLD.status IS DISTINCT FROM NEW.status)
    EXECUTE FUNCTION update_workflow_stats();

-- Update knowledge item usage statistics
CREATE OR REPLACE FUNCTION update_knowledge_usage()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE global_knowledge_items
        SET 
            usage_count = usage_count + 1,
            last_updated = NOW()
        WHERE id = NEW.knowledge_id::UUID;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Note: This trigger would be created when we have a feedback table
-- CREATE TRIGGER trigger_update_knowledge_usage
--     AFTER INSERT ON knowledge_feedback
--     FOR EACH ROW
--     EXECUTE FUNCTION update_knowledge_usage();

-- ============================================================================
-- INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- Insert default workflow templates
INSERT INTO workflow_executions (id, project_id, name, template_name, status, configuration) 
VALUES 
    (gen_random_uuid(), 'template-web-app', 'Web Application Template', 'web_application', 'template', 
     '{"description": "Standard web application development workflow", "phases": ["requirements", "architecture", "implementation", "testing", "deployment"]}'),
    (gen_random_uuid(), 'template-api-service', 'API Service Template', 'api_service', 'template',
     '{"description": "RESTful API service development workflow", "phases": ["requirements", "architecture", "implementation", "testing"]}'),
    (gen_random_uuid(), 'template-mobile-app', 'Mobile App Template', 'mobile_application', 'template',
     '{"description": "Mobile application development workflow", "phases": ["requirements", "ui_design", "implementation", "testing", "deployment"]}')
ON CONFLICT DO NOTHING;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO archon_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO archon_user;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE autonomous_teams IS 'Autonomous AI development teams with their composition and characteristics';
COMMENT ON TABLE team_agents IS 'Individual AI agents within autonomous teams with their capabilities and roles';
COMMENT ON TABLE workflow_executions IS 'Workflow orchestration instances for autonomous development processes';
COMMENT ON TABLE workflow_tasks IS 'Individual tasks within workflows with dependencies and status tracking';
COMMENT ON TABLE team_performance_profiles IS 'Performance profiles and analytics for autonomous teams';
COMMENT ON TABLE performance_data_points IS 'Time-series performance data for teams and agents';
COMMENT ON TABLE performance_optimizations IS 'Performance improvement recommendations and their implementation status';
COMMENT ON TABLE cross_project_metrics IS 'Project metrics collected for cross-project pattern analysis';
COMMENT ON TABLE identified_patterns IS 'Patterns and anti-patterns identified from cross-project analysis';
COMMENT ON TABLE knowledge_syntheses IS 'Comprehensive knowledge synthesis reports from cross-project learning';
COMMENT ON TABLE global_knowledge_items IS 'Knowledge items shared in the global autonomous teams network';
COMMENT ON TABLE network_nodes IS 'Organizations and teams participating in the global knowledge network';
COMMENT ON TABLE network_queries IS 'Knowledge network queries for analytics and caching optimization';