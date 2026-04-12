"""
config.py – Cấu hình tập trung cho pipeline tiền xử lý VCTK Corpus.
Tất cả đường dẫn và tham số được định nghĩa ở đây, không hardcode ở nơi khác.
"""

from pathlib import Path

# ─── Root của toàn bộ project ─────────────────────────────────────────────────
# src/preprocessing/config.py → lên 2 cấp → Beta/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
TARGET_SR       = 16_000    # Sample rate mục tiêu (Hz)
TARGET_DURATION = 5.0       # Độ dài cố định (giây)
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)  # = 80,000 samples

MIN_DURATION    = 1.0       # Thời lượng tối thiểu để file hợp lệ (giây)
MIN_RMS         = 1e-4      # Ngưỡng năng lượng – loại file silence

# ─── Sampling ─────────────────────────────────────────────────────────────────
TOTAL_FILES  = 500          # Tổng số file cần lấy
RANDOM_SEED  = 42           # Seed cố định → kết quả reproducible

# ─── Cột trong speaker-info.txt (0-indexed sau khi split()) ──────────────────
COL_SPEAKER_ID = 0
COL_GENDER     = 2          # 'F' hoặc 'M'
FEMALE_LABEL   = 'F'
