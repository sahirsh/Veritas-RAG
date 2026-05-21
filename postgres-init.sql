-- Runs only the first time Postgres initializes the database volume.
-- Ensures the pgvector extension is available as `vector`.
CREATE EXTENSION IF NOT EXISTS vector;
