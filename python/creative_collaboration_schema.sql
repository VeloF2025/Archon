-- Creative Collaboration Database Schema for Phase 10
-- Supabase PostgreSQL schema for Creative AI Collaboration features
-- Supports design-thinking AI partners, human-AI pair programming, 
-- collaborative design, and innovation acceleration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS creative_session_insights CASCADE;
DROP TABLE IF EXISTS innovation_metrics CASCADE;
DROP TABLE IF EXISTS breakthrough_indicators CASCADE;
DROP TABLE IF EXISTS cross_domain_insights CASCADE;
DROP TABLE IF EXISTS solution_concepts CASCADE;
DROP TABLE IF EXISTS design_elements CASCADE;
DROP TABLE IF EXISTS design_sessions CASCADE;
DROP TABLE IF EXISTS collaboration_interactions CASCADE;
DROP TABLE IF EXISTS collaboration_sessions CASCADE;
DROP TABLE IF EXISTS developer_profiles CASCADE;
DROP TABLE IF EXISTS session_contributions CASCADE;
DROP TABLE IF EXISTS creative_sessions CASCADE;
DROP TABLE IF EXISTS creative_problems CASCADE;

-- Creative Problems Table
CREATE TABLE creative_problems (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    domain VARCHAR(100) NOT NULL,
    complexity VARCHAR(20) DEFAULT 'moderate' CHECK (complexity IN ('simple', 'moderate', 'complex', 'highly_complex')),
    success_criteria TEXT[], -- Array of success criteria
    constraints TEXT[], -- Array of constraints
    stakeholders TEXT[], -- Array of stakeholders
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'solved', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Creative Sessions Table
CREATE TABLE creative_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id UUID NOT NULL REFERENCES creative_problems(id) ON DELETE CASCADE,
    session_type VARCHAR(50) NOT NULL DEFAULT 'brainstorming',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
    current_phase VARCHAR(30) DEFAULT 'ideation' CHECK (current_phase IN (
        'ideation', 'concept_development', 'prototyping', 'validation', 'synthesis', 'finalization'
    )),
    participants TEXT[], -- Array of participant identifiers
    duration_minutes INTEGER DEFAULT 60,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    session_config JSONB DEFAULT '{}'::jsonb,
    progress_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Session Contributions Table
CREATE TABLE session_contributions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES creative_sessions(id) ON DELETE CASCADE,
    contributor VARCHAR(255) NOT NULL, -- Human user or AI agent identifier
    contributor_type VARCHAR(20) DEFAULT 'human' CHECK (contributor_type IN ('human', 'ai_agent')),
    content TEXT NOT NULL,
    contribution_type VARCHAR(30) NOT NULL DEFAULT 'idea' CHECK (contribution_type IN (
        'idea', 'concept', 'feedback', 'question', 'solution', 'refinement'
    )),
    phase VARCHAR(30) NOT NULL,
    confidence_score FLOAT DEFAULT 0.7 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    impact_score FLOAT DEFAULT 0.5 CHECK (impact_score >= 0 AND impact_score <= 1),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Developer Profiles Table
CREATE TABLE developer_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    developer_id VARCHAR(100) UNIQUE NOT NULL,
    personality_type VARCHAR(30) NOT NULL CHECK (personality_type IN (
        'analytical', 'creative', 'pragmatic', 'collaborative', 'detail_oriented', 'big_picture'
    )),
    coding_style VARCHAR(30) NOT NULL CHECK (coding_style IN (
        'functional', 'object_oriented', 'procedural', 'agile', 'methodical', 'experimental'
    )),
    experience_level VARCHAR(20) NOT NULL CHECK (experience_level IN ('junior', 'mid', 'senior', 'expert')),
    preferred_languages TEXT[] NOT NULL, -- Array of programming languages
    interests TEXT[], -- Array of technical interests
    collaboration_preferences JSONB DEFAULT '{}'::jsonb,
    learning_style VARCHAR(30) DEFAULT 'adaptive',
    communication_style VARCHAR(30) DEFAULT 'adaptive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Collaboration Sessions Table
CREATE TABLE collaboration_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    developer_id VARCHAR(100) NOT NULL REFERENCES developer_profiles(developer_id) ON DELETE CASCADE,
    task_description TEXT NOT NULL,
    code_context TEXT,
    session_goals TEXT[], -- Array of session objectives
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed')),
    ai_personality JSONB DEFAULT '{}'::jsonb, -- Adapted AI personality for this session
    productivity_metrics JSONB DEFAULT '{}'::jsonb,
    learning_insights JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Collaboration Interactions Table
CREATE TABLE collaboration_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES collaboration_sessions(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL CHECK (interaction_type IN (
        'question', 'answer', 'suggestion', 'feedback', 'code_review', 'technical_discussion'
    )),
    content TEXT NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    response_type VARCHAR(50),
    response_content TEXT,
    response_metadata JSONB DEFAULT '{}'::jsonb,
    satisfaction_score FLOAT CHECK (satisfaction_score >= 1 AND satisfaction_score <= 10),
    helpfulness_score FLOAT CHECK (helpfulness_score >= 1 AND helpfulness_score <= 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Design Sessions Table
CREATE TABLE design_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_name VARCHAR(255) NOT NULL,
    design_brief TEXT NOT NULL,
    target_audience VARCHAR(255) NOT NULL,
    design_goals TEXT[] NOT NULL, -- Array of design objectives
    constraints TEXT[], -- Array of design constraints
    participants TEXT[], -- Array of session participants
    current_phase VARCHAR(30) DEFAULT 'research' CHECK (current_phase IN (
        'research', 'ideation', 'wireframing', 'visual_design', 'prototyping', 'testing', 'refinement', 'finalization'
    )),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed')),
    canvas_data JSONB DEFAULT '{}'::jsonb,
    design_system JSONB DEFAULT '{}'::jsonb,
    collaboration_analytics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Design Elements Table
CREATE TABLE design_elements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES design_sessions(id) ON DELETE CASCADE,
    element_type VARCHAR(50) NOT NULL CHECK (element_type IN (
        'text', 'button', 'input', 'image', 'container', 'navigation', 'form', 'chart', 'custom'
    )),
    content JSONB NOT NULL, -- Element properties and content
    position JSONB NOT NULL, -- Position coordinates and dimensions
    style JSONB DEFAULT '{}'::jsonb, -- Styling information
    version INTEGER DEFAULT 1,
    created_by VARCHAR(255) NOT NULL,
    ai_feedback JSONB DEFAULT '{}'::jsonb,
    interactions_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Solution Concepts Table (for innovation acceleration)
CREATE TABLE solution_concepts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id VARCHAR(100) NOT NULL, -- Links to innovation problems
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    approach VARCHAR(100) NOT NULL,
    innovation_type VARCHAR(30) NOT NULL CHECK (innovation_type IN (
        'breakthrough', 'incremental', 'disruptive', 'architectural', 'cross_domain', 'combinatorial'
    )),
    status VARCHAR(20) DEFAULT 'concept' CHECK (status IN ('concept', 'prototype', 'validated', 'implemented', 'scaled')),
    feasibility_score FLOAT NOT NULL CHECK (feasibility_score >= 0 AND feasibility_score <= 1),
    innovation_score FLOAT NOT NULL CHECK (innovation_score >= 0 AND innovation_score <= 1),
    risk_score FLOAT NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    potential_impact FLOAT NOT NULL CHECK (potential_impact >= 0 AND potential_impact <= 1),
    development_effort FLOAT NOT NULL CHECK (development_effort >= 0 AND development_effort <= 1),
    inspiration_domains TEXT[], -- Array of domains that inspired this solution
    key_insights TEXT[], -- Array of key insights
    implementation_steps TEXT[], -- Array of implementation steps
    success_metrics TEXT[], -- Array of success metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross Domain Insights Table
CREATE TABLE cross_domain_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_domain VARCHAR(100) NOT NULL,
    target_domain VARCHAR(100) NOT NULL,
    principle TEXT NOT NULL,
    example TEXT NOT NULL,
    applicability_score FLOAT NOT NULL CHECK (applicability_score >= 0 AND applicability_score <= 1),
    adaptation_notes TEXT NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.5 CHECK (success_rate >= 0 AND success_rate <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Breakthrough Indicators Table
CREATE TABLE breakthrough_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    solution_id UUID NOT NULL REFERENCES solution_concepts(id) ON DELETE CASCADE,
    indicator_type VARCHAR(50) NOT NULL CHECK (indicator_type IN (
        'performance_leap', 'paradigm_shift', 'market_disruption', 'technological_convergence', 
        'network_effects', 'overall_breakthrough_potential'
    )),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence TEXT[], -- Array of evidence supporting this indicator
    implications TEXT[], -- Array of implications
    validation_status VARCHAR(20) DEFAULT 'pending' CHECK (validation_status IN ('pending', 'confirmed', 'refuted')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Innovation Metrics Table
CREATE TABLE innovation_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id VARCHAR(100) NOT NULL,
    solution_id UUID NOT NULL REFERENCES solution_concepts(id) ON DELETE CASCADE,
    novelty_score FLOAT NOT NULL CHECK (novelty_score >= 0 AND novelty_score <= 1),
    usefulness_score FLOAT NOT NULL CHECK (usefulness_score >= 0 AND usefulness_score <= 1),
    elegance_score FLOAT NOT NULL CHECK (elegance_score >= 0 AND elegance_score <= 1),
    scalability_score FLOAT NOT NULL CHECK (scalability_score >= 0 AND scalability_score <= 1),
    sustainability_score FLOAT NOT NULL CHECK (sustainability_score >= 0 AND sustainability_score <= 1),
    market_potential FLOAT NOT NULL CHECK (market_potential >= 0 AND market_potential <= 1),
    technical_feasibility FLOAT NOT NULL CHECK (technical_feasibility >= 0 AND technical_feasibility <= 1),
    overall_innovation_score FLOAT NOT NULL CHECK (overall_innovation_score >= 0 AND overall_innovation_score <= 1),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metrics_version VARCHAR(10) DEFAULT 'v1.0',
    calculation_method JSONB DEFAULT '{}'::jsonb
);

-- Creative Session Insights Table (analytics and derived insights)
CREATE TABLE creative_session_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES creative_sessions(id) ON DELETE CASCADE,
    insight_type VARCHAR(50) NOT NULL CHECK (insight_type IN (
        'participation_balance', 'idea_categories', 'creative_momentum', 'collaboration_patterns', 'success_indicators'
    )),
    insight_data JSONB NOT NULL,
    confidence_level FLOAT DEFAULT 0.7 CHECK (confidence_level >= 0 AND confidence_level <= 1),
    impact_level VARCHAR(20) DEFAULT 'medium' CHECK (impact_level IN ('low', 'medium', 'high', 'critical')),
    recommendations TEXT[],
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for optimal performance
CREATE INDEX idx_creative_problems_domain ON creative_problems(domain);
CREATE INDEX idx_creative_problems_status ON creative_problems(status);
CREATE INDEX idx_creative_problems_created_at ON creative_problems(created_at);

CREATE INDEX idx_creative_sessions_problem_id ON creative_sessions(problem_id);
CREATE INDEX idx_creative_sessions_status ON creative_sessions(status);
CREATE INDEX idx_creative_sessions_phase ON creative_sessions(current_phase);
CREATE INDEX idx_creative_sessions_started_at ON creative_sessions(started_at);

CREATE INDEX idx_session_contributions_session_id ON session_contributions(session_id);
CREATE INDEX idx_session_contributions_type ON session_contributions(contribution_type);
CREATE INDEX idx_session_contributions_created_at ON session_contributions(created_at);

CREATE INDEX idx_developer_profiles_developer_id ON developer_profiles(developer_id);
CREATE INDEX idx_developer_profiles_personality ON developer_profiles(personality_type);
CREATE INDEX idx_developer_profiles_experience ON developer_profiles(experience_level);

CREATE INDEX idx_collaboration_sessions_developer_id ON collaboration_sessions(developer_id);
CREATE INDEX idx_collaboration_sessions_status ON collaboration_sessions(status);
CREATE INDEX idx_collaboration_sessions_started_at ON collaboration_sessions(started_at);

CREATE INDEX idx_collaboration_interactions_session_id ON collaboration_interactions(session_id);
CREATE INDEX idx_collaboration_interactions_type ON collaboration_interactions(interaction_type);
CREATE INDEX idx_collaboration_interactions_created_at ON collaboration_interactions(created_at);

CREATE INDEX idx_design_sessions_status ON design_sessions(status);
CREATE INDEX idx_design_sessions_phase ON design_sessions(current_phase);
CREATE INDEX idx_design_sessions_created_at ON design_sessions(created_at);

CREATE INDEX idx_design_elements_session_id ON design_elements(session_id);
CREATE INDEX idx_design_elements_type ON design_elements(element_type);
CREATE INDEX idx_design_elements_created_by ON design_elements(created_by);

CREATE INDEX idx_solution_concepts_problem_id ON solution_concepts(problem_id);
CREATE INDEX idx_solution_concepts_innovation_type ON solution_concepts(innovation_type);
CREATE INDEX idx_solution_concepts_status ON solution_concepts(status);
CREATE INDEX idx_solution_concepts_innovation_score ON solution_concepts(innovation_score);
CREATE INDEX idx_solution_concepts_created_at ON solution_concepts(created_at);

CREATE INDEX idx_breakthrough_indicators_solution_id ON breakthrough_indicators(solution_id);
CREATE INDEX idx_breakthrough_indicators_type ON breakthrough_indicators(indicator_type);
CREATE INDEX idx_breakthrough_indicators_confidence ON breakthrough_indicators(confidence);

CREATE INDEX idx_innovation_metrics_solution_id ON innovation_metrics(solution_id);
CREATE INDEX idx_innovation_metrics_overall_score ON innovation_metrics(overall_innovation_score);
CREATE INDEX idx_innovation_metrics_calculated_at ON innovation_metrics(calculated_at);

CREATE INDEX idx_creative_session_insights_session_id ON creative_session_insights(session_id);
CREATE INDEX idx_creative_session_insights_type ON creative_session_insights(insight_type);
CREATE INDEX idx_creative_session_insights_impact ON creative_session_insights(impact_level);

-- Composite indexes for complex queries
CREATE INDEX idx_sessions_problem_status ON creative_sessions(problem_id, status);
CREATE INDEX idx_contributions_session_phase ON session_contributions(session_id, phase);
CREATE INDEX idx_solutions_problem_score ON solution_concepts(problem_id, innovation_score DESC);
CREATE INDEX idx_interactions_session_type ON collaboration_interactions(session_id, interaction_type);

-- GIN indexes for JSONB columns (for efficient JSON queries)
CREATE INDEX idx_creative_problems_metadata_gin ON creative_problems USING GIN (metadata);
CREATE INDEX idx_creative_sessions_config_gin ON creative_sessions USING GIN (session_config);
CREATE INDEX idx_creative_sessions_progress_gin ON creative_sessions USING GIN (progress_data);
CREATE INDEX idx_design_sessions_canvas_gin ON design_sessions USING GIN (canvas_data);
CREATE INDEX idx_design_elements_content_gin ON design_elements USING GIN (content);
CREATE INDEX idx_collaboration_sessions_personality_gin ON collaboration_sessions USING GIN (ai_personality);
CREATE INDEX idx_creative_session_insights_data_gin ON creative_session_insights USING GIN (insight_data);

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_creative_problems_updated_at BEFORE UPDATE ON creative_problems FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_creative_sessions_updated_at BEFORE UPDATE ON creative_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_session_contributions_updated_at BEFORE UPDATE ON session_contributions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_developer_profiles_updated_at BEFORE UPDATE ON developer_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_collaboration_sessions_updated_at BEFORE UPDATE ON collaboration_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_collaboration_interactions_updated_at BEFORE UPDATE ON collaboration_interactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_design_sessions_updated_at BEFORE UPDATE ON design_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_design_elements_updated_at BEFORE UPDATE ON design_elements FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_solution_concepts_updated_at BEFORE UPDATE ON solution_concepts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cross_domain_insights_updated_at BEFORE UPDATE ON cross_domain_insights FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common analytics queries

-- Creative Session Analytics View
CREATE VIEW creative_session_analytics AS
SELECT 
    cp.domain,
    cs.status,
    cs.current_phase,
    COUNT(*) as session_count,
    AVG(EXTRACT(EPOCH FROM (COALESCE(cs.completed_at, NOW()) - cs.started_at))/60) as avg_duration_minutes,
    COUNT(sc.id) as total_contributions,
    COUNT(DISTINCT sc.contributor) as unique_contributors,
    AVG(sc.confidence_score) as avg_confidence,
    AVG(sc.impact_score) as avg_impact
FROM creative_sessions cs
JOIN creative_problems cp ON cs.problem_id = cp.id
LEFT JOIN session_contributions sc ON cs.id = sc.session_id
GROUP BY cp.domain, cs.status, cs.current_phase;

-- Developer Collaboration Performance View
CREATE VIEW developer_collaboration_performance AS
SELECT 
    dp.developer_id,
    dp.personality_type,
    dp.experience_level,
    COUNT(cs.id) as total_sessions,
    AVG(EXTRACT(EPOCH FROM (COALESCE(cs.ended_at, NOW()) - cs.started_at))/60) as avg_session_duration_minutes,
    COUNT(ci.id) as total_interactions,
    AVG(ci.satisfaction_score) as avg_satisfaction,
    AVG(ci.helpfulness_score) as avg_helpfulness,
    COUNT(ci.id)::float / NULLIF(COUNT(cs.id), 0) as interactions_per_session
FROM developer_profiles dp
LEFT JOIN collaboration_sessions cs ON dp.developer_id = cs.developer_id
LEFT JOIN collaboration_interactions ci ON cs.id = ci.session_id
GROUP BY dp.developer_id, dp.personality_type, dp.experience_level;

-- Innovation Pipeline Performance View
CREATE VIEW innovation_pipeline_performance AS
SELECT 
    sc.problem_id,
    sc.innovation_type,
    COUNT(*) as solution_count,
    AVG(sc.innovation_score) as avg_innovation_score,
    AVG(sc.feasibility_score) as avg_feasibility_score,
    AVG(sc.potential_impact) as avg_potential_impact,
    COUNT(bi.id) as breakthrough_indicators_count,
    AVG(bi.confidence) as avg_breakthrough_confidence,
    COUNT(CASE WHEN im.overall_innovation_score > 0.8 THEN 1 END) as high_potential_solutions
FROM solution_concepts sc
LEFT JOIN breakthrough_indicators bi ON sc.id = bi.solution_id
LEFT JOIN innovation_metrics im ON sc.id = im.solution_id
GROUP BY sc.problem_id, sc.innovation_type;

-- Design Collaboration Effectiveness View
CREATE VIEW design_collaboration_effectiveness AS
SELECT 
    ds.project_name,
    ds.current_phase,
    ds.status,
    COUNT(de.id) as total_elements,
    COUNT(DISTINCT de.created_by) as unique_contributors,
    AVG(de.version) as avg_element_iterations,
    COUNT(CASE WHEN de.ai_feedback IS NOT NULL THEN 1 END) as elements_with_ai_feedback,
    EXTRACT(EPOCH FROM (COALESCE(ds.updated_at, NOW()) - ds.created_at))/86400 as project_age_days
FROM design_sessions ds
LEFT JOIN design_elements de ON ds.id = de.session_id
GROUP BY ds.id, ds.project_name, ds.current_phase, ds.status, ds.created_at, ds.updated_at;

-- Row Level Security (RLS) policies for multi-tenancy (optional)
-- These would be enabled based on specific tenant requirements

-- ALTER TABLE creative_problems ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE creative_sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE developer_profiles ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE design_sessions ENABLE ROW LEVEL SECURITY;

-- Sample RLS policy (would need to be customized based on auth system):
-- CREATE POLICY creative_problems_tenant_isolation ON creative_problems
--     FOR ALL USING (metadata->>'tenant_id' = current_setting('app.current_tenant')::text);

-- Insert sample data for testing (optional)
-- This can be uncommented for development/testing environments

-- INSERT INTO creative_problems (title, description, domain, success_criteria, constraints, stakeholders) VALUES
-- ('AI-Powered Code Review System', 'Design an intelligent code review system that provides contextual feedback', 'software_engineering', 
--  ARRAY['Accurate feedback', 'Fast processing', 'Developer adoption'], 
--  ARRAY['Integration with existing tools', 'Privacy requirements'], 
--  ARRAY['Development team', 'Engineering managers', 'Quality assurance']);

-- INSERT INTO developer_profiles (developer_id, personality_type, coding_style, experience_level, preferred_languages, interests) VALUES
-- ('dev_alice_001', 'analytical', 'functional', 'senior', ARRAY['Python', 'TypeScript', 'Rust'], ARRAY['AI/ML', 'System architecture']);

-- INSERT INTO design_sessions (project_name, design_brief, target_audience, design_goals, constraints, participants) VALUES
-- ('Mobile Banking App Redesign', 'Create intuitive and secure mobile banking experience', 'Banking customers', 
--  ARRAY['Improve usability', 'Enhance security', 'Increase engagement'], 
--  ARRAY['Regulatory compliance', 'Accessibility standards'], 
--  ARRAY['ui_designer@bank.com', 'ux_researcher@bank.com']);

COMMENT ON SCHEMA public IS 'Creative Collaboration Schema for Phase 10 - Supports design-thinking AI partners, human-AI pair programming, collaborative design, and innovation acceleration';
COMMENT ON TABLE creative_problems IS 'Complex problems requiring creative AI collaboration and innovative solutions';
COMMENT ON TABLE creative_sessions IS 'Active creative collaboration sessions with human and AI participants';
COMMENT ON TABLE session_contributions IS 'Individual contributions (ideas, concepts, feedback) within creative sessions';
COMMENT ON TABLE developer_profiles IS 'Developer personality profiles for personalized AI collaboration';
COMMENT ON TABLE collaboration_sessions IS 'Human-AI pair programming and collaboration sessions';
COMMENT ON TABLE collaboration_interactions IS 'Individual interactions within collaboration sessions';
COMMENT ON TABLE design_sessions IS 'Collaborative design sessions with real-time co-design capabilities';
COMMENT ON TABLE design_elements IS 'Individual design elements on collaborative design canvases';
COMMENT ON TABLE solution_concepts IS 'Generated solution concepts from innovation acceleration engine';
COMMENT ON TABLE cross_domain_insights IS 'Insights and analogies drawn from other domains';
COMMENT ON TABLE breakthrough_indicators IS 'Indicators suggesting breakthrough innovation potential';
COMMENT ON TABLE innovation_metrics IS 'Comprehensive innovation quality metrics for solutions';
COMMENT ON TABLE creative_session_insights IS 'Derived analytics and insights from creative sessions';