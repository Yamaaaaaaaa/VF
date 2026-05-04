"""
config.py – Cấu hình tập trung cho pipeline tiền xử lý VCTK Corpus.
Tất cả đường dẫn và tham số được định nghĩa ở đây, không hardcode ở nơi khác.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Root của toàn bộ project ─────────────────────────────────────────────────
# src/preprocessing/config.py → lên 2 cấp → Beta/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load file .env từ root project
load_dotenv(PROJECT_ROOT / ".env")

# ─── Nguồn dữ liệu: VCTK Corpus gốc ──────────────────────────────────────────
VCTK_ROOT    = PROJECT_ROOT / "data" / "raw" / "VCTK-Corpus" / "VCTK-Corpus"
SPEAKER_INFO = VCTK_ROOT / "speaker-info.txt"
WAV48_DIR    = VCTK_ROOT / "wav48"

# ─── Output sau tiền xử lý ────────────────────────────────────────────────────
# Theo P9: data/raw/<speaker>/<file>.wav
PROCESSED_DIR = PROJECT_ROOT / "data" / "raw"
# Theo P9: data/features/<speaker>/<file>.npy  (dùng cho feature extraction)
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"

# ─── Tham số tiền xử lý âm thanh ─────────────────────────────────────────────
TARGET_SR       = int(os.getenv("TARGET_SR", 16000))
TARGET_DURATION = float(os.getenv("TARGET_DURATION", 5.0))
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)

MIN_DURATION    = float(os.getenv("MIN_DURATION", 1.0))
MIN_RMS         = float(os.getenv("MIN_RMS", 1e-4))

# ─── Sampling ─────────────────────────────────────────────────────────────────
TOTAL_FILES  = int(os.getenv("TOTAL_FILES", 500))
RANDOM_SEED  = int(os.getenv("RANDOM_SEED", 42))

# ─── Database Config ──────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "voice_db")
DB_USER     = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin_password")

# ─── Cột trong speaker-info.txt (0-indexed sau khi split()) ──────────────────
COL_SPEAKER_ID = 0
COL_GENDER     = 2          # 'F' hoặc 'M'
FEMALE_LABEL   = 'F'
