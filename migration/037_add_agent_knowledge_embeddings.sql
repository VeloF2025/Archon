-- Migration: Add agent knowledge embeddings table for vector similarity search
-- Requires: pgvector extension in Supabase

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the agent knowledge table with vector embeddings
CREATE TABLE IF NOT EXISTS archon_agent_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES archon_agents_v3(id) ON DELETE CASCADE,
    
    -- Knowledge content and context
    content TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    
    -- Vector embedding for similarity search (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384) NOT NULL,
    
    -- Metadata and tracking
    metadata JSONB DEFAULT '{}',
    relevance_score FLOAT DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX idx_agent_knowledge_agent_id ON archon_agent_knowledge(agent_id);
CREATE INDEX idx_agent_knowledge_relevance ON archon_agent_knowledge(relevance_score DESC);
CREATE INDEX idx_agent_knowledge_created_at ON archon_agent_knowledge(created_at DESC);

-- Create vector similarity search index using ivfflat
-- This enables fast similarity searches across embeddings
CREATE INDEX idx_agent_knowledge_embedding ON archon_agent_knowledge 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create function to search for similar knowledge items
CREATE OR REPLACE FUNCTION search_similar_knowledge(
    query_embedding vector(384),
    agent_id_filter UUID DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    agent_id UUID,
    content TEXT,
    context JSONB,
    relevance_score FLOAT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        k.id,
        k.agent_id,
        k.content,
        k.context,
        k.relevance_score,
        1 - (k.embedding <=> query_embedding) AS similarity
    FROM archon_agent_knowledge k
    WHERE 
        (agent_id_filter IS NULL OR k.agent_id = agent_id_filter)
        AND 1 - (k.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY k.embedding <=> query_embedding
    LIMIT max_results;
END;
$$;

-- Create function to update knowledge access metrics
CREATE OR REPLACE FUNCTION update_knowledge_access(knowledge_id UUID)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE archon_agent_knowledge
    SET 
        access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE id = knowledge_id;
END;
$$;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_knowledge_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_knowledge_updated_at
    BEFORE UPDATE ON archon_agent_knowledge
    FOR EACH ROW
    EXECUTE FUNCTION update_knowledge_updated_at();

-- Add RLS policies for security
ALTER TABLE archon_agent_knowledge ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can do everything
CREATE POLICY "Service role full access" ON archon_agent_knowledge
    FOR ALL
    USING (auth.role() = 'service_role');

-- Policy: Agents can only access their own knowledge and global knowledge
CREATE POLICY "Agent knowledge access" ON archon_agent_knowledge
    FOR SELECT
    USING (
        auth.uid()::UUID = agent_id 
        OR agent_id IN (
            SELECT id FROM archon_agents_v3 
            WHERE configuration->>'is_global_knowledge' = 'true'
        )
    );

-- Create view for agent knowledge statistics
CREATE OR REPLACE VIEW agent_knowledge_stats AS
SELECT 
    agent_id,
    COUNT(*) as total_items,
    AVG(relevance_score) as avg_relevance,
    SUM(access_count) as total_accesses,
    MAX(created_at) as latest_item_at,
    MAX(last_accessed) as last_accessed_at
FROM archon_agent_knowledge
GROUP BY agent_id;

-- Add comment for documentation
COMMENT ON TABLE archon_agent_knowledge IS 'Stores vector embeddings of agent knowledge for similarity-based retrieval and learning';
COMMENT ON COLUMN archon_agent_knowledge.embedding IS 'Vector embedding (384 dimensions) for similarity search using all-MiniLM-L6-v2 model';
COMMENT ON FUNCTION search_similar_knowledge IS 'Searches for similar knowledge items using cosine similarity on vector embeddings';