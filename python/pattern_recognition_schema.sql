-- Pattern Recognition Engine Database Schema
-- Part of Archon Enhancement 2025 - Phase 1

-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Code patterns table
CREATE TABLE IF NOT EXISTS code_patterns (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    language TEXT NOT NULL,
    signature TEXT UNIQUE NOT NULL,
    embedding vector(1536),  -- OpenAI embedding dimension
    examples JSONB DEFAULT '[]',
    frequency INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.0,
    effectiveness_score REAL DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    is_antipattern BOOLEAN DEFAULT FALSE,
    performance_impact TEXT,
    suggested_alternative TEXT,
    project_id TEXT REFERENCES archon_projects(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_patterns_embedding 
ON code_patterns USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_patterns_language 
ON code_patterns(language);

CREATE INDEX IF NOT EXISTS idx_patterns_category 
ON code_patterns(category);

CREATE INDEX IF NOT EXISTS idx_patterns_project 
ON code_patterns(project_id);

CREATE INDEX IF NOT EXISTS idx_patterns_antipattern 
ON code_patterns(is_antipattern);

CREATE INDEX IF NOT EXISTS idx_patterns_effectiveness 
ON code_patterns(effectiveness_score DESC);

CREATE INDEX IF NOT EXISTS idx_patterns_usage 
ON code_patterns(usage_count DESC);

-- Pattern relationships table
CREATE TABLE IF NOT EXISTS pattern_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern1_id TEXT NOT NULL REFERENCES code_patterns(id) ON DELETE CASCADE,
    pattern2_id TEXT NOT NULL REFERENCES code_patterns(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL, -- 'alternative', 'complementary', 'conflicting', 'prerequisite'
    strength REAL DEFAULT 0.0 CHECK (strength >= 0 AND strength <= 1),
    evidence_count INTEGER DEFAULT 0,
    contexts JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pattern1_id, pattern2_id)
);

CREATE INDEX IF NOT EXISTS idx_relationships_pattern1 
ON pattern_relationships(pattern1_id);

CREATE INDEX IF NOT EXISTS idx_relationships_pattern2 
ON pattern_relationships(pattern2_id);

CREATE INDEX IF NOT EXISTS idx_relationships_type 
ON pattern_relationships(relationship_type);

-- Pattern usage history table
CREATE TABLE IF NOT EXISTS pattern_usage_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id TEXT NOT NULL REFERENCES code_patterns(id) ON DELETE CASCADE,
    project_id TEXT REFERENCES archon_projects(id) ON DELETE SET NULL,
    user_id TEXT,
    context TEXT,
    was_effective BOOLEAN,
    was_useful BOOLEAN,
    feedback TEXT,
    code_before TEXT,
    code_after TEXT,
    metrics JSONB DEFAULT '{}',  -- Performance metrics, quality scores, etc.
    used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_pattern 
ON pattern_usage_history(pattern_id);

CREATE INDEX IF NOT EXISTS idx_usage_project 
ON pattern_usage_history(project_id);

CREATE INDEX IF NOT EXISTS idx_usage_user 
ON pattern_usage_history(user_id);

CREATE INDEX IF NOT EXISTS idx_usage_date 
ON pattern_usage_history(used_at DESC);

-- Pattern detection events table (for Kafka integration)
CREATE TABLE IF NOT EXISTS pattern_detection_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL, -- 'detection', 'recommendation', 'feedback', 'analysis'
    pattern_ids TEXT[] DEFAULT '{}',
    project_id TEXT,
    source_file TEXT,
    language TEXT,
    event_data JSONB DEFAULT '{}',
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_type 
ON pattern_detection_events(event_type);

CREATE INDEX IF NOT EXISTS idx_events_processed 
ON pattern_detection_events(processed);

CREATE INDEX IF NOT EXISTS idx_events_created 
ON pattern_detection_events(created_at DESC);

-- Pattern learning metrics table
CREATE TABLE IF NOT EXISTS pattern_learning_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id TEXT NOT NULL REFERENCES code_patterns(id) ON DELETE CASCADE,
    metric_type TEXT NOT NULL, -- 'accuracy', 'precision', 'recall', 'f1_score'
    metric_value REAL,
    sample_size INTEGER,
    evaluation_date DATE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pattern_id, metric_type, evaluation_date)
);

CREATE INDEX IF NOT EXISTS idx_metrics_pattern 
ON pattern_learning_metrics(pattern_id);

CREATE INDEX IF NOT EXISTS idx_metrics_type 
ON pattern_learning_metrics(metric_type);

CREATE INDEX IF NOT EXISTS idx_metrics_date 
ON pattern_learning_metrics(evaluation_date DESC);

-- Functions for pattern operations

-- Function to update pattern effectiveness based on feedback
CREATE OR REPLACE FUNCTION update_pattern_effectiveness(
    p_pattern_id TEXT,
    p_was_effective BOOLEAN
) RETURNS VOID AS $$
DECLARE
    v_current_score REAL;
    v_usage_count INTEGER;
    v_new_score REAL;
BEGIN
    SELECT effectiveness_score, usage_count 
    INTO v_current_score, v_usage_count
    FROM code_patterns
    WHERE id = p_pattern_id;
    
    IF FOUND THEN
        -- Calculate new effectiveness using weighted average
        IF p_was_effective THEN
            v_new_score := (v_current_score * v_usage_count + 1.0) / (v_usage_count + 1);
        ELSE
            v_new_score := (v_current_score * v_usage_count) / (v_usage_count + 1);
        END IF;
        
        -- Update pattern
        UPDATE code_patterns
        SET effectiveness_score = v_new_score,
            usage_count = v_usage_count + 1,
            last_used = NOW(),
            updated_at = NOW()
        WHERE id = p_pattern_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar patterns using vector similarity
CREATE OR REPLACE FUNCTION find_similar_patterns(
    p_embedding vector(1536),
    p_language TEXT DEFAULT NULL,
    p_category TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 10,
    p_threshold REAL DEFAULT 0.7
) RETURNS TABLE (
    pattern_id TEXT,
    pattern_name TEXT,
    similarity REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cp.id,
        cp.name,
        1 - (cp.embedding <=> p_embedding) AS similarity
    FROM code_patterns cp
    WHERE 
        (p_language IS NULL OR cp.language = p_language)
        AND (p_category IS NULL OR cp.category = p_category)
        AND cp.embedding IS NOT NULL
        AND 1 - (cp.embedding <=> p_embedding) >= p_threshold
    ORDER BY cp.embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_code_patterns_updated_at
    BEFORE UPDATE ON code_patterns
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pattern_relationships_updated_at
    BEFORE UPDATE ON pattern_relationships
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- View for pattern statistics
CREATE OR REPLACE VIEW pattern_statistics AS
SELECT 
    cp.language,
    cp.category,
    COUNT(*) as pattern_count,
    AVG(cp.effectiveness_score) as avg_effectiveness,
    AVG(cp.confidence) as avg_confidence,
    SUM(cp.usage_count) as total_usage,
    COUNT(CASE WHEN cp.is_antipattern THEN 1 END) as antipattern_count
FROM code_patterns cp
GROUP BY cp.language, cp.category;

-- View for top patterns by language
CREATE OR REPLACE VIEW top_patterns_by_language AS
SELECT 
    cp.*,
    ROW_NUMBER() OVER (PARTITION BY cp.language ORDER BY cp.effectiveness_score DESC, cp.usage_count DESC) as rank
FROM code_patterns cp
WHERE cp.is_antipattern = FALSE;

-- Grant permissions (adjust as needed)
GRANT SELECT, INSERT, UPDATE ON code_patterns TO authenticated;
GRANT SELECT, INSERT, UPDATE ON pattern_relationships TO authenticated;
GRANT SELECT, INSERT ON pattern_usage_history TO authenticated;
GRANT SELECT, INSERT ON pattern_detection_events TO authenticated;
GRANT SELECT, INSERT ON pattern_learning_metrics TO authenticated;
GRANT SELECT ON pattern_statistics TO authenticated;
GRANT SELECT ON top_patterns_by_language TO authenticated;

-- Add RLS policies (if using Row Level Security)
ALTER TABLE code_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_usage_history ENABLE ROW LEVEL SECURITY;

-- Policy for reading patterns (everyone can read)
CREATE POLICY "Patterns are viewable by everyone" 
ON code_patterns FOR SELECT 
USING (true);

-- Policy for inserting patterns (authenticated users only)
CREATE POLICY "Authenticated users can insert patterns" 
ON code_patterns FOR INSERT 
WITH CHECK (auth.role() = 'authenticated');

-- Policy for updating patterns (authenticated users only)
CREATE POLICY "Authenticated users can update patterns" 
ON code_patterns FOR UPDATE 
USING (auth.role() = 'authenticated');

-- Comments for documentation
COMMENT ON TABLE code_patterns IS 'Stores detected code patterns with embeddings for similarity search';
COMMENT ON TABLE pattern_relationships IS 'Tracks relationships between different patterns';
COMMENT ON TABLE pattern_usage_history IS 'Historical record of pattern usage and effectiveness';
COMMENT ON TABLE pattern_detection_events IS 'Event stream for pattern detection system';
COMMENT ON TABLE pattern_learning_metrics IS 'ML metrics for pattern detection accuracy';
COMMENT ON COLUMN code_patterns.embedding IS 'Vector embedding for semantic similarity search';
COMMENT ON COLUMN code_patterns.effectiveness_score IS 'Weighted average of pattern effectiveness based on feedback';
COMMENT ON COLUMN pattern_relationships.relationship_type IS 'Type of relationship: alternative, complementary, conflicting, or prerequisite';