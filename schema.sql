CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    source VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Index configured for cosine distance (<=>)
CREATE INDEX IF NOT EXISTS idx_feedback_embedding_hnsw 
ON feedback USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);