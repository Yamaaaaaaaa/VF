-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng lưu trữ metadata và vector đặc trưng
CREATE TABLE IF NOT EXISTS voice_records (
    file_id      SERIAL PRIMARY KEY,
    speaker      TEXT NOT NULL,
    accent       TEXT,
    gender       TEXT CHECK (gender IN ('male', 'female')),
    age          INTEGER,
    file_path    TEXT NOT NULL,
    npy_path     TEXT NOT NULL,
    duration_sec REAL,
    sample_rate  INTEGER,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    embedding    VECTOR(99)
);

-- Index metadata
CREATE INDEX IF NOT EXISTS idx_voice_gender  ON voice_records (gender);
CREATE INDEX IF NOT EXISTS idx_voice_speaker ON voice_records (speaker);
CREATE INDEX IF NOT EXISTS idx_voice_accent  ON voice_records (accent);

-- Index pgvector ANN (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_voice_embedding_ivfflat
    ON voice_records
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 22);
