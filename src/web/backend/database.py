import psycopg2
from psycopg2.extras import DictCursor
import numpy as np

import os

# Database connection settings
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "voice_db"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD", "admin_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def fetch_records(limit=20, offset=0, search_query="", gender_filter="", accent_filter=""):
    """
    Fetch records from DB with pagination and filters.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = "SELECT file_id, speaker, accent, gender, age, file_path, duration_sec FROM voice_records WHERE 1=1"
            params = []
            
            if search_query:
                query += " AND speaker ILIKE %s"
                params.append(f"%{search_query}%")
                
            if gender_filter:
                query += " AND gender = %s"
                params.append(gender_filter)
                
            if accent_filter:
                query += " AND accent ILIKE %s"
                params.append(f"%{accent_filter}%")
                
            query += " ORDER BY file_id DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            records = cur.fetchall()
            
            # Count total
            count_query = "SELECT COUNT(*) FROM voice_records WHERE 1=1"
            count_params = []
            if search_query:
                count_query += " AND speaker ILIKE %s"
                count_params.append(f"%{search_query}%")
            if gender_filter:
                count_query += " AND gender = %s"
                count_params.append(gender_filter)
            if accent_filter:
                count_query += " AND accent ILIKE %s"
                count_params.append(f"%{accent_filter}%")
            
            cur.execute(count_query, count_params)
            total = cur.fetchone()[0]
            
            return [dict(r) for r in records], total
    finally:
        conn.close()

def search_similar_voices(query_embedding: np.ndarray, top_k=5):
    """
    Search for top_k similar voices using vector cosine similarity.
    Requires query_embedding to be a 99-dimensional list or np.ndarray.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # Format numpy array to pgvector string format: '[vi, vj, ...]'
            embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
            
            # Dùng cosine distance ( <=> ) với pgvector
            query = """
                SELECT file_id, speaker, accent, gender, age, file_path, duration_sec, 
                       1 - (embedding <=> %s::vector) AS similarity 
                FROM voice_records 
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
            """
            cur.execute(query, (embedding_str, embedding_str, top_k))
            results = cur.fetchall()
            return [dict(r) for r in results]
    finally:
        conn.close()


def fetch_embedding(file_id: int):
    """Return (file_path, embedding_list) for a given file_id."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                "SELECT file_path, embedding::text FROM voice_records WHERE file_id = %s",
                (file_id,)
            )
            row = cur.fetchone()
            if not row:
                return None, None
            import re
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', row['embedding'])
            embedding = [float(x) for x in nums]
            return row['file_path'], embedding
    finally:
        conn.close()


def search_by_embedding(embedding: list, top_k: int = 5, exclude_file_id: int = -1):
    """Similarity search using a pre-fetched embedding list; excludes the source record."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            emb_str = "[" + ",".join(map(str, embedding)) + "]"
            query = """
                SELECT file_id, speaker, accent, gender, age, file_path, duration_sec,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM voice_records
                WHERE file_id != %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(query, (emb_str, exclude_file_id, emb_str, top_k))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

