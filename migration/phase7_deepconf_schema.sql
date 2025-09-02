-- =====================================================
-- Phase 7 DeepConf Integration Database Schema Migration
-- =====================================================
-- Date: 2025-09-01
-- Purpose: Implement comprehensive confidence scoring and performance metrics
--          for Dynamic Confidence System integration with Archon
-- Version: 1.0
-- Author: Database Architect (via Claude Code)
-- =====================================================

-- Begin transaction for atomic migration
BEGIN;

-- =====================================================
-- SECTION 1: CONFIDENCE SCORES TABLE
-- =====================================================

-- Main confidence scores table for storing dynamic confidence assessments
CREATE TABLE IF NOT EXISTS archon_confidence_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    
    -- Core confidence metrics (0.0000 to 1.0000 scale)
    factual_confidence DECIMAL(5,4) NOT NULL CHECK (factual_confidence >= 0.0000 AND factual_confidence <= 1.0000),
    reasoning_confidence DECIMAL(5,4) NOT NULL CHECK (reasoning_confidence >= 0.0000 AND reasoning_confidence <= 1.0000),
    contextual_relevance DECIMAL(5,4) NOT NULL CHECK (contextual_relevance >= 0.0000 AND contextual_relevance <= 1.0000),
    
    -- Uncertainty bounds for confidence intervals
    uncertainty_lower DECIMAL(5,4) NOT NULL CHECK (uncertainty_lower >= 0.0000 AND uncertainty_lower <= 1.0000),
    uncertainty_upper DECIMAL(5,4) NOT NULL CHECK (uncertainty_upper >= 0.0000 AND uncertainty_upper <= 1.0000),
    
    -- Composite confidence score (calculated)
    overall_confidence DECIMAL(5,4) GENERATED ALWAYS AS (
        (factual_confidence + reasoning_confidence + contextual_relevance) / 3.0
    ) STORED,
    
    -- Model consensus data (JSON structure for multiple model agreements)
    model_consensus JSONB NOT NULL DEFAULT '{}',
    
    -- Additional metadata
    model_version VARCHAR(100),
    temperature DECIMAL(3,2),
    max_tokens INTEGER,
    prompt_hash VARCHAR(64), -- SHA-256 hash for cache optimization
    
    -- Request context
    user_id UUID,
    session_id UUID,
    request_type VARCHAR(50), -- 'rag_query', 'chat', 'code_generation', etc.
    
    -- Audit fields following Archon pattern
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_uncertainty_bounds CHECK (uncertainty_lower <= uncertainty_upper),
    CONSTRAINT chk_confidence_valid CHECK (
        factual_confidence IS NOT NULL AND 
        reasoning_confidence IS NOT NULL AND 
        contextual_relevance IS NOT NULL
    )
);

-- =====================================================
-- SECTION 2: PERFORMANCE METRICS TABLE
-- =====================================================

-- Performance metrics for system monitoring and optimization
CREATE TABLE IF NOT EXISTS archon_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Timestamp with millisecond precision for accurate performance tracking
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metric_date DATE GENERATED ALWAYS AS (DATE(timestamp)) STORED,
    
    -- Core performance metrics (0.0000 to 1.0000 scale except response_time_ms)
    token_efficiency DECIMAL(5,4) NOT NULL CHECK (token_efficiency >= 0.0000 AND token_efficiency <= 1.0000),
    response_time_ms INTEGER NOT NULL CHECK (response_time_ms >= 0),
    confidence_accuracy DECIMAL(5,4) NOT NULL CHECK (confidence_accuracy >= 0.0000 AND confidence_accuracy <= 1.0000),
    hallucination_rate DECIMAL(5,4) NOT NULL CHECK (hallucination_rate >= 0.0000 AND hallucination_rate <= 1.0000),
    system_load DECIMAL(5,4) NOT NULL CHECK (system_load >= 0.0000),
    
    -- Additional performance indicators
    memory_usage_mb INTEGER CHECK (memory_usage_mb >= 0),
    cpu_usage_percent DECIMAL(5,2) CHECK (cpu_usage_percent >= 0.0000 AND cpu_usage_percent <= 100.0000),
    cache_hit_rate DECIMAL(5,4) CHECK (cache_hit_rate >= 0.0000 AND cache_hit_rate <= 1.0000),
    
    -- Request categorization
    request_type VARCHAR(50),
    model_version VARCHAR(100),
    endpoint VARCHAR(255),
    
    -- Quality metrics
    user_satisfaction_score DECIMAL(3,2) CHECK (user_satisfaction_score >= 1.00 AND user_satisfaction_score <= 5.00),
    error_rate DECIMAL(5,4) CHECK (error_rate >= 0.0000 AND error_rate <= 1.0000),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- SECTION 3: CONFIDENCE CALIBRATION TABLE
-- =====================================================

-- Table for tracking confidence calibration data for model improvement
CREATE TABLE IF NOT EXISTS archon_confidence_calibration (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    confidence_score_id UUID NOT NULL,
    
    -- Actual outcome validation
    actual_accuracy DECIMAL(5,4) CHECK (actual_accuracy >= 0.0000 AND actual_accuracy <= 1.0000),
    predicted_confidence DECIMAL(5,4) NOT NULL CHECK (predicted_confidence >= 0.0000 AND predicted_confidence <= 1.0000),
    calibration_error DECIMAL(5,4) GENERATED ALWAYS AS (
        ABS(actual_accuracy - predicted_confidence)
    ) STORED,
    
    -- Validation method
    validation_method VARCHAR(50) NOT NULL, -- 'human_review', 'automated_test', 'cross_validation'
    validator_id UUID,
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Feedback data
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5),
    feedback_comments TEXT,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key constraint
    CONSTRAINT fk_confidence_calibration_score 
        FOREIGN KEY (confidence_score_id) 
        REFERENCES archon_confidence_scores(id) 
        ON DELETE CASCADE
);

-- =====================================================
-- SECTION 4: INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Confidence Scores Indexes
CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_request_id 
    ON archon_confidence_scores(request_id);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_created_at 
    ON archon_confidence_scores(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_overall_confidence 
    ON archon_confidence_scores(overall_confidence DESC);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_user_session 
    ON archon_confidence_scores(user_id, session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_request_type 
    ON archon_confidence_scores(request_type);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_prompt_hash 
    ON archon_confidence_scores(prompt_hash);

-- GIN index for model_consensus JSONB queries
CREATE INDEX IF NOT EXISTS idx_archon_confidence_scores_consensus_gin 
    ON archon_confidence_scores USING gin (model_consensus);

-- Performance Metrics Indexes  
CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_timestamp 
    ON archon_performance_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_date 
    ON archon_performance_metrics(metric_date DESC);

CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_request_type 
    ON archon_performance_metrics(request_type);

CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_response_time 
    ON archon_performance_metrics(response_time_ms);

CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_endpoint 
    ON archon_performance_metrics(endpoint);

-- Composite index for time-series queries
CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_composite 
    ON archon_performance_metrics(request_type, timestamp DESC, response_time_ms);

-- GIN index for metadata JSONB queries
CREATE INDEX IF NOT EXISTS idx_archon_performance_metrics_metadata_gin 
    ON archon_performance_metrics USING gin (metadata);

-- Confidence Calibration Indexes
CREATE INDEX IF NOT EXISTS idx_archon_confidence_calibration_score_id 
    ON archon_confidence_calibration(confidence_score_id);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_calibration_validation 
    ON archon_confidence_calibration(validation_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_archon_confidence_calibration_method 
    ON archon_confidence_calibration(validation_method);

-- =====================================================
-- SECTION 5: UPDATE TRIGGERS
-- =====================================================

-- Update trigger for confidence scores table
CREATE TRIGGER update_archon_confidence_scores_updated_at
    BEFORE UPDATE ON archon_confidence_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- SECTION 6: VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for confidence trends analysis
CREATE OR REPLACE VIEW archon_confidence_trends AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    request_type,
    COUNT(*) as request_count,
    AVG(overall_confidence) as avg_confidence,
    MIN(overall_confidence) as min_confidence,
    MAX(overall_confidence) as max_confidence,
    STDDEV(overall_confidence) as confidence_stddev,
    AVG(factual_confidence) as avg_factual,
    AVG(reasoning_confidence) as avg_reasoning,
    AVG(contextual_relevance) as avg_contextual
FROM archon_confidence_scores
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), request_type
ORDER BY hour DESC, request_type;

-- View for performance dashboard
CREATE OR REPLACE VIEW archon_performance_dashboard AS
SELECT 
    DATE_TRUNC('minute', timestamp) as minute,
    request_type,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    AVG(token_efficiency) as avg_token_efficiency,
    AVG(confidence_accuracy) as avg_confidence_accuracy,
    AVG(hallucination_rate) as avg_hallucination_rate,
    AVG(system_load) as avg_system_load,
    AVG(COALESCE(cache_hit_rate, 0)) as avg_cache_hit_rate
FROM archon_performance_metrics
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('minute', timestamp), request_type
ORDER BY minute DESC, request_type;

-- View for calibration analysis
CREATE OR REPLACE VIEW archon_calibration_analysis AS
SELECT 
    DATE_TRUNC('day', c.validation_timestamp) as day,
    c.validation_method,
    COUNT(*) as calibration_count,
    AVG(c.calibration_error) as avg_calibration_error,
    AVG(c.actual_accuracy) as avg_actual_accuracy,
    AVG(cs.overall_confidence) as avg_predicted_confidence,
    STDDEV(c.calibration_error) as calibration_error_stddev
FROM archon_confidence_calibration c
JOIN archon_confidence_scores cs ON c.confidence_score_id = cs.id
WHERE c.validation_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', c.validation_timestamp), c.validation_method
ORDER BY day DESC, c.validation_method;

-- =====================================================
-- SECTION 7: FUNCTIONS FOR CONFIDENCE OPERATIONS
-- =====================================================

-- Function to calculate confidence statistics
CREATE OR REPLACE FUNCTION calculate_confidence_stats(
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW() - INTERVAL '24 hours',
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    p_request_type VARCHAR DEFAULT NULL
) RETURNS TABLE (
    total_requests BIGINT,
    avg_confidence DECIMAL(5,4),
    median_confidence DECIMAL(5,4),
    confidence_range DECIMAL(5,4),
    low_confidence_count BIGINT,
    high_confidence_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_requests,
        AVG(overall_confidence)::DECIMAL(5,4) as avg_confidence,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY overall_confidence)::DECIMAL(5,4) as median_confidence,
        (MAX(overall_confidence) - MIN(overall_confidence))::DECIMAL(5,4) as confidence_range,
        COUNT(CASE WHEN overall_confidence < 0.5 THEN 1 END)::BIGINT as low_confidence_count,
        COUNT(CASE WHEN overall_confidence >= 0.8 THEN 1 END)::BIGINT as high_confidence_count
    FROM archon_confidence_scores
    WHERE created_at BETWEEN p_start_date AND p_end_date
    AND (p_request_type IS NULL OR request_type = p_request_type);
END;
$$ LANGUAGE plpgsql;

-- Function to insert performance metrics with validation
CREATE OR REPLACE FUNCTION insert_performance_metric(
    p_token_efficiency DECIMAL(5,4),
    p_response_time_ms INTEGER,
    p_confidence_accuracy DECIMAL(5,4),
    p_hallucination_rate DECIMAL(5,4),
    p_system_load DECIMAL(5,4),
    p_request_type VARCHAR DEFAULT NULL,
    p_model_version VARCHAR DEFAULT NULL,
    p_endpoint VARCHAR DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
    v_metric_id UUID;
BEGIN
    -- Validate input parameters
    IF p_token_efficiency < 0 OR p_token_efficiency > 1 THEN
        RAISE EXCEPTION 'token_efficiency must be between 0.0000 and 1.0000';
    END IF;
    
    IF p_response_time_ms < 0 THEN
        RAISE EXCEPTION 'response_time_ms must be non-negative';
    END IF;
    
    -- Insert metric record
    INSERT INTO archon_performance_metrics (
        token_efficiency,
        response_time_ms,
        confidence_accuracy,
        hallucination_rate,
        system_load,
        request_type,
        model_version,
        endpoint,
        metadata
    ) VALUES (
        p_token_efficiency,
        p_response_time_ms,
        p_confidence_accuracy,
        p_hallucination_rate,
        p_system_load,
        p_request_type,
        p_model_version,
        p_endpoint,
        p_metadata
    ) RETURNING id INTO v_metric_id;
    
    RETURN v_metric_id;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SECTION 8: ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on confidence scores
ALTER TABLE archon_confidence_scores ENABLE ROW LEVEL SECURITY;

-- Policy for confidence scores - users can only see their own scores
CREATE POLICY confidence_scores_user_policy ON archon_confidence_scores
    FOR ALL USING (
        user_id = auth.uid() OR 
        auth.jwt() ->> 'role' = 'admin' OR
        user_id IS NULL  -- Allow anonymous queries for system metrics
    );

-- Enable RLS on performance metrics (admin-only access)
ALTER TABLE archon_performance_metrics ENABLE ROW LEVEL SECURITY;

-- Policy for performance metrics - admin access only
CREATE POLICY performance_metrics_admin_policy ON archon_performance_metrics
    FOR ALL USING (auth.jwt() ->> 'role' = 'admin');

-- Enable RLS on calibration data
ALTER TABLE archon_confidence_calibration ENABLE ROW LEVEL SECURITY;

-- Policy for calibration data - admin and validators only
CREATE POLICY calibration_validator_policy ON archon_confidence_calibration
    FOR ALL USING (
        validator_id = auth.uid() OR 
        auth.jwt() ->> 'role' IN ('admin', 'validator')
    );

-- =====================================================
-- SECTION 9: COMMENTS AND DOCUMENTATION
-- =====================================================

-- Table comments
COMMENT ON TABLE archon_confidence_scores IS 'Dynamic confidence scores for AI responses with uncertainty quantification and model consensus tracking';
COMMENT ON TABLE archon_performance_metrics IS 'System performance metrics for monitoring token efficiency, response times, and quality indicators';
COMMENT ON TABLE archon_confidence_calibration IS 'Confidence calibration data for model improvement and accuracy validation';

-- Column comments for confidence scores
COMMENT ON COLUMN archon_confidence_scores.request_id IS 'Unique identifier linking to the original request';
COMMENT ON COLUMN archon_confidence_scores.factual_confidence IS 'Confidence in factual accuracy (0.0000-1.0000)';
COMMENT ON COLUMN archon_confidence_scores.reasoning_confidence IS 'Confidence in reasoning quality (0.0000-1.0000)';
COMMENT ON COLUMN archon_confidence_scores.contextual_relevance IS 'Relevance to provided context (0.0000-1.0000)';
COMMENT ON COLUMN archon_confidence_scores.uncertainty_lower IS 'Lower bound of confidence interval (0.0000-1.0000)';
COMMENT ON COLUMN archon_confidence_scores.uncertainty_upper IS 'Upper bound of confidence interval (0.0000-1.0000)';
COMMENT ON COLUMN archon_confidence_scores.overall_confidence IS 'Computed average of core confidence metrics';
COMMENT ON COLUMN archon_confidence_scores.model_consensus IS 'JSON data containing multi-model agreement scores';
COMMENT ON COLUMN archon_confidence_scores.prompt_hash IS 'SHA-256 hash for caching and deduplication';

-- Column comments for performance metrics
COMMENT ON COLUMN archon_performance_metrics.token_efficiency IS 'Ratio of useful tokens to total tokens used';
COMMENT ON COLUMN archon_performance_metrics.response_time_ms IS 'End-to-end response time in milliseconds';
COMMENT ON COLUMN archon_performance_metrics.confidence_accuracy IS 'Accuracy of confidence predictions vs actual outcomes';
COMMENT ON COLUMN archon_performance_metrics.hallucination_rate IS 'Rate of detected hallucinations in responses';
COMMENT ON COLUMN archon_performance_metrics.system_load IS 'Current system load factor (can exceed 1.0)';

-- =====================================================
-- SECTION 10: VALIDATION AND VERIFICATION
-- =====================================================

-- Verify tables were created successfully
DO $$ 
DECLARE
    table_count INTEGER;
    index_count INTEGER;
    view_count INTEGER;
    function_count INTEGER;
BEGIN 
    -- Count tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables 
    WHERE table_name IN ('archon_confidence_scores', 'archon_performance_metrics', 'archon_confidence_calibration')
    AND table_schema = 'public';
    
    -- Count indexes
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE tablename LIKE 'archon_confidence%' OR tablename LIKE 'archon_performance%';
    
    -- Count views
    SELECT COUNT(*) INTO view_count
    FROM information_schema.views
    WHERE table_name LIKE 'archon_%confidence%' OR table_name LIKE 'archon_%performance%';
    
    -- Count functions
    SELECT COUNT(*) INTO function_count
    FROM information_schema.routines
    WHERE routine_name LIKE '%confidence%' OR routine_name LIKE '%performance%';
    
    -- Validation results
    IF table_count = 3 THEN
        RAISE NOTICE '✓ All 3 DeepConf tables created successfully';
    ELSE
        RAISE EXCEPTION '✗ Expected 3 tables, found %', table_count;
    END IF;
    
    IF index_count >= 15 THEN
        RAISE NOTICE '✓ Performance indexes created successfully (% indexes)', index_count;
    ELSE
        RAISE NOTICE '⚠ Found % indexes, expected at least 15', index_count;
    END IF;
    
    IF view_count = 3 THEN
        RAISE NOTICE '✓ All 3 analysis views created successfully';
    ELSE
        RAISE NOTICE '⚠ Found % views, expected 3', view_count;
    END IF;
    
    IF function_count >= 2 THEN
        RAISE NOTICE '✓ DeepConf functions created successfully (% functions)', function_count;
    ELSE
        RAISE NOTICE '⚠ Found % functions, expected at least 2', function_count;
    END IF;
    
    RAISE NOTICE '🎉 Phase 7 DeepConf Integration schema migration completed successfully!';
END $$;

-- Commit the transaction
COMMIT;

-- =====================================================
-- ROLLBACK SCRIPT (Save separately or run manually)
-- =====================================================
/*
-- ROLLBACK INSTRUCTIONS:
-- To rollback this migration, run the following commands:

BEGIN;

-- Drop views
DROP VIEW IF EXISTS archon_confidence_trends CASCADE;
DROP VIEW IF EXISTS archon_performance_dashboard CASCADE;
DROP VIEW IF EXISTS archon_calibration_analysis CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS calculate_confidence_stats(TIMESTAMP WITH TIME ZONE, TIMESTAMP WITH TIME ZONE, VARCHAR);
DROP FUNCTION IF EXISTS insert_performance_metric(DECIMAL, INTEGER, DECIMAL, DECIMAL, DECIMAL, VARCHAR, VARCHAR, VARCHAR, JSONB);

-- Drop tables (in reverse dependency order)
DROP TABLE IF EXISTS archon_confidence_calibration CASCADE;
DROP TABLE IF EXISTS archon_performance_metrics CASCADE;
DROP TABLE IF EXISTS archon_confidence_scores CASCADE;

COMMIT;

-- Verify rollback
SELECT 'Rollback completed - DeepConf tables removed' as status;
*/