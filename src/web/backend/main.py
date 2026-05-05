from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os

from database import fetch_records, search_similar_voices, fetch_embedding, search_by_embedding
from audio_processor import process_audio_file, analyze_from_path

app = FastAPI(title="Voice Similarity API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve data directory (works both locally and inside Docker /data volume)
data_path = os.environ.get(
    "DATA_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))
)

# ── Pydantic models ────────────────────────────────────────────────────────────

class RecordResponse(BaseModel):
    file_id: int
    speaker: str
    accent: Optional[str] = None
    gender: str
    age: Optional[int] = None
    file_path: str
    duration_sec: Optional[float] = None

class PaginatedRecords(BaseModel):
    total: int
    records: List[RecordResponse]

class SearchResultResponse(BaseModel):
    file_id: int
    speaker: str
    accent: Optional[str] = None
    gender: str
    age: Optional[int] = None
    file_path: str
    duration_sec: Optional[float] = None
    similarity: float

# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/records", response_model=PaginatedRecords)
def get_records(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: str = Query("", description="Search by speaker ID"),
    gender: str = Query("", description="Filter by gender"),
    accent: str = Query("", description="Filter by accent")
):
    try:
        records, total = fetch_records(limit, offset, search, gender, accent)
        return {"total": total, "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=List[SearchResultResponse])
async def search_voice(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    try:
        file_bytes = await file.read()
        embedding = process_audio_file(file_bytes)
        results = search_similar_voices(embedding, top_k=top_k)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.get("/api/records/{file_id}/analyze")
def analyze_record(file_id: int):
    """Return waveform, mel spectrogram, MFCC matrix and 99D embedding for a stored record."""
    file_path, embedding = fetch_embedding(file_id)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Record not found")
    # Normalize Windows backslashes for Linux container filesystem
    normalized = file_path.replace("\\", "/")
    full_path = os.path.join(data_path, normalized)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {full_path}")
    analysis = analyze_from_path(full_path)
    analysis["embedding"] = embedding
    return analysis


@app.get("/api/records/{file_id}/similar")
def similar_records(file_id: int, top_k: int = Query(5, ge=1, le=10)):
    """Return top-k similar records for a given file_id (excludes itself)."""
    file_path, embedding = fetch_embedding(file_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Record not found")
    results = search_by_embedding(embedding, top_k=top_k, exclude_file_id=file_id)
    return results


# ── Static file serving ────────────────────────────────────────────────────────
# IMPORTANT: mount AFTER all API routes so it doesn't shadow them
if os.path.exists(data_path):
    app.mount("/data", StaticFiles(directory=data_path), name="data")
