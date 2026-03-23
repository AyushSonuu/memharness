-- memharness PostgreSQL + pgvector schema
-- Full schema creation with vector support for all 10 memory types
-- Copyright (c) 2026 Ayush Sonuu
-- Licensed under MIT License

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =========================================================================
-- Conversational Memory Table (SQL-based)
-- =========================================================================
CREATE TABLE IF NOT EXISTS conversational_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key TEXT NOT NULL UNIQUE,
    namespace TEXT[] NOT NULL DEFAULT '{}',
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    summary_id UUID
);

-- Composite index on thread_id and created_at for efficient conversation retrieval
CREATE INDEX IF NOT EXISTS idx_conv_thread_time
ON conversational_memory(thread_id, created_at DESC);

-- GIN index on namespace array for containment queries
CREATE INDEX IF NOT EXISTS idx_conv_namespace
ON conversational_memory USING GIN(namespace);

-- Index on summary_id for finding original messages from summaries
CREATE INDEX IF NOT EXISTS idx_conv_summary
ON conversational_memory(summary_id) WHERE summary_id IS NOT NULL;

-- Index on key for fast lookups
CREATE INDEX IF NOT EXISTS idx_conv_key
ON conversational_memory(key);

-- =========================================================================
-- Tool Log Memory Table (SQL-based audit trail)
-- =========================================================================
CREATE TABLE IF NOT EXISTS tool_log_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key TEXT NOT NULL UNIQUE,
    namespace TEXT[] NOT NULL DEFAULT '{}',
    thread_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Composite index for retrieving tool logs by thread
CREATE INDEX IF NOT EXISTS idx_tool_log_thread_time
ON tool_log_memory(thread_id, created_at DESC);

-- Index on tool_name for filtering by specific tools
CREATE INDEX IF NOT EXISTS idx_tool_log_tool
ON tool_log_memory(tool_name);

-- GIN index on namespace
CREATE INDEX IF NOT EXISTS idx_tool_log_namespace
ON tool_log_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_tool_log_key
ON tool_log_memory(key);

-- =========================================================================
-- Summary Memory Table (Vector-based)
-- Note: Vector dimension will be substituted at runtime (default: 768)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS summary_memory is defined via Python (vector_dim param)
-- Placeholder for vector dimension: {vector_dim}

-- HNSW index for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_summary_embedding
ON summary_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_summary_thread
ON summary_memory(thread_id);

CREATE INDEX IF NOT EXISTS idx_summary_namespace
ON summary_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_summary_key
ON summary_memory(key);

-- Trigram index for text similarity in hybrid search
CREATE INDEX IF NOT EXISTS idx_summary_content_trgm
ON summary_memory USING GIN(content gin_trgm_ops);

-- =========================================================================
-- Knowledge Base Memory Table (Vector-based)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS knowledge_base_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_kb_embedding
ON knowledge_base_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_kb_source
ON knowledge_base_memory(source);

CREATE INDEX IF NOT EXISTS idx_kb_namespace
ON knowledge_base_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_kb_key
ON knowledge_base_memory(key);

-- Trigram index for text similarity
CREATE INDEX IF NOT EXISTS idx_kb_content_trgm
ON knowledge_base_memory USING GIN(content gin_trgm_ops);

-- =========================================================================
-- Entity Memory Table (Vector-based)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS entity_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_entity_embedding
ON entity_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_entity_type
ON entity_memory(entity_type);

CREATE INDEX IF NOT EXISTS idx_entity_namespace
ON entity_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_entity_key
ON entity_memory(key);

CREATE INDEX IF NOT EXISTS idx_entity_content_trgm
ON entity_memory USING GIN(content gin_trgm_ops);

-- =========================================================================
-- Workflow Memory Table (Vector-based)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS workflow_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_workflow_embedding
ON workflow_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_workflow_namespace
ON workflow_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_workflow_key
ON workflow_memory(key);

CREATE INDEX IF NOT EXISTS idx_workflow_content_trgm
ON workflow_memory USING GIN(content gin_trgm_ops);

-- =========================================================================
-- Toolbox Memory Table (Vector-based with VFS path)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS toolbox_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_toolbox_embedding
ON toolbox_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_toolbox_tool_name
ON toolbox_memory(tool_name);

CREATE INDEX IF NOT EXISTS idx_toolbox_vfs_path
ON toolbox_memory(vfs_path);

CREATE INDEX IF NOT EXISTS idx_toolbox_namespace
ON toolbox_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_toolbox_key
ON toolbox_memory(key);

-- =========================================================================
-- Skills Memory Table (Vector-based)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS skills_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_skills_embedding
ON skills_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_skills_name
ON skills_memory(skill_name);

CREATE INDEX IF NOT EXISTS idx_skills_namespace
ON skills_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_skills_key
ON skills_memory(key);

CREATE INDEX IF NOT EXISTS idx_skills_content_trgm
ON skills_memory USING GIN(content gin_trgm_ops);

-- =========================================================================
-- File Memory Table (Hybrid - Vector + metadata)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS file_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_file_embedding
ON file_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_file_path
ON file_memory(file_path);

CREATE INDEX IF NOT EXISTS idx_file_hash
ON file_memory(file_hash);

CREATE INDEX IF NOT EXISTS idx_file_namespace
ON file_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_file_key
ON file_memory(key);

-- =========================================================================
-- Persona Memory Table (Vector-based)
-- =========================================================================
-- CREATE TABLE IF NOT EXISTS persona_memory is defined via Python (vector_dim param)

CREATE INDEX IF NOT EXISTS idx_persona_embedding
ON persona_memory USING hnsw(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_persona_name
ON persona_memory(persona_name);

CREATE INDEX IF NOT EXISTS idx_persona_namespace
ON persona_memory USING GIN(namespace);

CREATE INDEX IF NOT EXISTS idx_persona_key
ON persona_memory(key);

CREATE INDEX IF NOT EXISTS idx_persona_content_trgm
ON persona_memory USING GIN(content gin_trgm_ops);

-- =========================================================================
-- Foreign Key Constraints
-- =========================================================================
-- Add foreign key from conversational to summary after both tables exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_conv_summary'
    ) THEN
        ALTER TABLE conversational_memory
        ADD CONSTRAINT fk_conv_summary
        FOREIGN KEY (summary_id) REFERENCES summary_memory(id)
        ON DELETE SET NULL;
    END IF;
END $$;
