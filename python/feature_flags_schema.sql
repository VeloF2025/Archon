-- Feature Flags Schema for Archon
-- Provides runtime feature toggle capabilities for gradual rollouts and A/B testing

-- Main feature flags table
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    is_enabled BOOLEAN DEFAULT false,
    rollout_percentage DECIMAL(5,2) DEFAULT 0.00 CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    targeting_rules JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    environment VARCHAR(50) DEFAULT 'production'
);

-- Feature flag variants for A/B testing
CREATE TABLE IF NOT EXISTS feature_flag_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_id UUID REFERENCES feature_flags(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    value JSONB NOT NULL,
    weight DECIMAL(5,2) DEFAULT 0.00 CHECK (weight >= 0 AND weight <= 100),
    is_control BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(flag_id, name)
);

-- User feature flag assignments for consistent experience
CREATE TABLE IF NOT EXISTS user_feature_assignments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_id UUID REFERENCES feature_flags(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    variant_id UUID REFERENCES feature_flag_variants(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    environment VARCHAR(50) DEFAULT 'production',
    UNIQUE(flag_id, user_id, environment)
);

-- Feature flag evaluation logs for analytics
CREATE TABLE IF NOT EXISTS feature_flag_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    flag_id UUID REFERENCES feature_flags(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    variant_id UUID REFERENCES feature_flag_variants(id) ON DELETE SET NULL,
    evaluation_result JSONB NOT NULL,
    context JSONB DEFAULT '{}',
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    environment VARCHAR(50) DEFAULT 'production'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_flags_name ON feature_flags(name);
CREATE INDEX IF NOT EXISTS idx_feature_flags_enabled ON feature_flags(is_enabled) WHERE is_enabled = true;
CREATE INDEX IF NOT EXISTS idx_feature_flags_environment ON feature_flags(environment);

CREATE INDEX IF NOT EXISTS idx_feature_flag_variants_flag_id ON feature_flag_variants(flag_id);
CREATE INDEX IF NOT EXISTS idx_feature_flag_variants_weight ON feature_flag_variants(weight) WHERE weight > 0;

CREATE INDEX IF NOT EXISTS idx_user_assignments_flag_user ON user_feature_assignments(flag_id, user_id);
CREATE INDEX IF NOT EXISTS idx_user_assignments_environment ON user_feature_assignments(environment);

CREATE INDEX IF NOT EXISTS idx_evaluations_flag_id ON feature_flag_evaluations(flag_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_user_id ON feature_flag_evaluations(user_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON feature_flag_evaluations(evaluated_at);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_feature_flags_updated_at 
    BEFORE UPDATE ON feature_flags 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert some default feature flags for Archon
INSERT INTO feature_flags (name, description, is_enabled, rollout_percentage) VALUES
('projects_feature', 'Enable project management functionality', true, 100.00),
('youtube_integration', 'Enable YouTube content analysis integration', true, 100.00),
('advanced_rag', 'Enable advanced RAG with multi-vector search', false, 0.00),
('deepconf_scoring', 'Enable DeepConf confidence scoring for decisions', true, 100.00),
('parallel_execution', 'Enable parallel agent execution for faster processing', true, 100.00),
('tdd_enforcement', 'Enable strict TDD enforcement gates', true, 100.00),
('claude_code_bridge', 'Enable Claude Code integration bridge', true, 100.00),
('web_intelligence_tools', 'Enable enhanced web intelligence gathering tools', false, 25.00)
ON CONFLICT (name) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE feature_flags IS 'Runtime feature toggles for gradual rollouts and A/B testing';
COMMENT ON TABLE feature_flag_variants IS 'Variants for A/B testing and multivariate experiments';
COMMENT ON TABLE user_feature_assignments IS 'Consistent user assignments to feature variants';
COMMENT ON TABLE feature_flag_evaluations IS 'Audit log of feature flag evaluations for analytics';

COMMENT ON COLUMN feature_flags.targeting_rules IS 'JSON rules for user/context-based targeting';
COMMENT ON COLUMN feature_flags.metadata IS 'Additional metadata like tags, owner, etc.';
COMMENT ON COLUMN feature_flag_variants.weight IS 'Percentage weight for random assignment (0-100)';
COMMENT ON COLUMN feature_flag_evaluations.evaluation_result IS 'Full evaluation result including variant and metadata';
COMMENT ON COLUMN feature_flag_evaluations.context IS 'User context during evaluation (location, device, etc.)';