"""
preprocess.py – Pipeline tiền xử lý VCTK Corpus.

Pipeline 5 bước (theo P3 trong Claude.md):
  1. Đọc speaker-info.txt → lọc female speakers
  2. Duyệt tất cả .wav của female speakers trong wav48/
  3. Kiểm tra chất lượng file (loại silence, quá ngắn, bị lỗi)
  4. Resample 48kHz → 16kHz + cắt/pad về đúng 5 giây
  5. Sampling 500 file (phân bổ đều theo speaker) → lưu ra data/raw/<speaker>/

Chạy:
    python -m src.preprocessing.preprocess
    hoặc:
    python src/preprocessing/preprocess.py
"""

import random
import shutil
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from config import (
    SPEAKER_INFO, WAV48_DIR, PROCESSED_DIR,
    TARGET_SR, TARGET_SAMPLES, MIN_DURATION, MIN_RMS,
    TOTAL_FILES, RANDOM_SEED,
    COL_SPEAKER_ID, COL_GENDER, FEMALE_LABEL,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Đọc & lọc metadata
# ══════════════════════════════════════════════════════════════════════════════

def load_female_speakers(speaker_info_path: Path) -> list[str]:
    """
    Đọc speaker-info.txt, trả về list speaker folder name của giọng nữ.
    Định dạng file: ID  AGE  GENDER  ACCENTS  REGION
    Ví dụ kết quả : ['p225', 'p228', 'p229', ...]
    """
    female_speakers: list[str] = []

    with open(speaker_info_path, "r", encoding="utf-8") as f:
        next(f)  # Bỏ dòng header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            speaker_id = parts[COL_SPEAKER_ID]   # VD: "225"
            gender     = parts[COL_GENDER]        # VD: "F"

            if gender == FEMALE_LABEL:
                female_speakers.append(f"p{speaker_id}")  # → "p225"

    log.info(f"[Bước 1] Tìm thấy {len(female_speakers)} female speakers.")
    return female_speakers


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Duyệt file .wav của female speakers
# ══════════════════════════════════════════════════════════════════════════════

def collect_wav_files(
    wav48_dir: Path,
    female_speakers: list[str],
) -> dict[str, list[Path]]:
    """
    Với mỗi female speaker, lấy toàn bộ file .wav trong wav48/<speaker>/.
    Trả về dict: { 'p225': [Path(...), ...], 'p228': [...], ... }
    """
    files_by_speaker: dict[str, list[Path]] = {}

    for speaker in female_speakers:
        speaker_dir = wav48_dir / speaker
        if not speaker_dir.exists():
            log.warning(f"  Không tìm thấy thư mục: {speaker_dir}")
            continue

        wav_files = sorted(speaker_dir.glob("*.wav"))
        if wav_files:
            files_by_speaker[speaker] = wav_files

    total = sum(len(v) for v in files_by_speaker.values())
    log.info(
        f"[Bước 2] Thu thập xong: {total} file .wav "
        f"từ {len(files_by_speaker)} speakers."
    )
    return files_by_speaker


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: Kiểm tra chất lượng file
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_audio(wav_path: Path) -> bool:
    """
    Trả về True nếu file âm thanh đạt yêu cầu:
      - Đọc được (không bị corrupt)
      - Thời lượng >= MIN_DURATION giây
      - Năng lượng RMS > MIN_RMS (không phải silence)
    """
    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        duration = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2)))
        return duration >= MIN_DURATION and rms > MIN_RMS
    except Exception:
        return False


def filter_valid_files(
    files_by_speaker: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    """
    Lọc bỏ file corrupt, silence, hoặc quá ngắn.
    Hiển thị progress bar trên toàn bộ file cần kiểm tra.
    """
    all_files = [
        (speaker, f)
        for speaker, files in files_by_speaker.items()
        for f in files
    ]

    valid_by_speaker: dict[str, list[Path]] = defaultdict(list)
    rejected = 0

    for speaker, wav_path in tqdm(all_files, desc="[Bước 3] Kiểm tra chất lượng"):
        if is_valid_audio(wav_path):
            valid_by_speaker[speaker].append(wav_path)
        else:
            rejected += 1

    total_valid = sum(len(v) for v in valid_by_speaker.values())
    log.info(
        f"[Bước 3] Hợp lệ: {total_valid} file. "
        f"Loại bỏ: {rejected} file (lỗi / silence / quá ngắn)."
    )
    return dict(valid_by_speaker)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: Resample + Cắt/Pad
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_audio(wav_path: Path) -> np.ndarray:
    """
    Load → resample 48kHz → 16kHz → cắt hoặc pad về đúng 5 giây.

    Returns:
        np.ndarray shape (80000,) – float32, normalized
    """
    # librosa.load tự resample từ sr gốc về TARGET_SR
    y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

    if len(y) >= TARGET_SAMPLES:
        # Cắt lấy 5 giây đầu (nội dung quan trọng thường ở đầu câu)
        y = y[:TARGET_SAMPLES]
    else:
        # Pad zero ở cuối để đủ 5 giây
        pad_len = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")

    return y  # shape: (80_000,), float32


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: Sampling 500 file (phân bổ đều theo speaker)
# ══════════════════════════════════════════════════════════════════════════════

def sample_files(
    valid_files_by_speaker: dict[str, list[Path]],
    total: int = TOTAL_FILES,
    seed: int = RANDOM_SEED,
) -> list[tuple[str, Path]]:
    """
    Phân bổ đều: mỗi speaker lấy khoảng (total // n_speakers) file.
    Nếu speaker không đủ quota → lấy hết và bù từ speaker khác.

    Returns:
        list[(speaker_id, wav_path)] đã được chọn, đủ `total` phần tử.
    """
    random.seed(seed)
    n_speakers = len(valid_files_by_speaker)
    quota      = total // n_speakers  # ~7-8 file/speaker cho ~65 speakers

    selected: list[tuple[str, Path]] = []
    leftover_pool: list[tuple[str, Path]] = []

    for speaker, files in valid_files_by_speaker.items():
        shuffled = files[:]
        random.shuffle(shuffled)
        # Lấy đủ quota hoặc tất cả nếu không đủ
        selected.extend((speaker, f) for f in shuffled[:quota])
        leftover_pool.extend((speaker, f) for f in shuffled[quota:])

    # Bù nếu chưa đủ TOTAL_FILES
    deficit = total - len(selected)
    if deficit > 0 and leftover_pool:
        extra = random.sample(leftover_pool, min(deficit, len(leftover_pool)))
        selected.extend(extra)

    # Shuffle cuối để thứ tự không bị nhóm theo speaker
    random.shuffle(selected)

    log.info(
        f"[Bước 5] Đã chọn {len(selected)} file "
        f"từ {n_speakers} speakers (quota ≈ {quota}/speaker)."
    )
    return selected[:total]


# ══════════════════════════════════════════════════════════════════════════════
# LƯU VÀ TỔNG KẾT
# ══════════════════════════════════════════════════════════════════════════════

def save_processed_files(
    selected: list[tuple[str, Path]],
    output_dir: Path,
) -> list[dict]:
    """
    Với mỗi file đã chọn:
      1. Tiền xử lý (resample + cắt/pad)
      2. Lưu ra output_dir/<speaker>/<filename>.wav
      3. Trả về list metadata record để insert DB sau.

    Output structure (theo P9):
        data/raw/
        ├── p225/
        │   ├── p225_001.wav
        │   └── p225_002.wav
        └── p228/
            └── p228_005.wav
    """
    records: list[dict] = []

    for speaker, src_path in tqdm(selected, desc="[Lưu] Tiền xử lý & ghi file"):
        # Tạo thư mục đích nếu chưa có
        dest_dir = output_dir / speaker
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / src_path.name

        # Bỏ qua nếu file đã tồn tại (chạy lại không overwrite)
        if dest_path.exists():
            continue

        try:
            y = preprocess_audio(src_path)
            # Lưu bằng soundfile: PCM 16-bit để tiết kiệm dung lượng
            sf.write(str(dest_path), y, TARGET_SR, subtype="PCM_16")

            records.append({
                "speaker"   : speaker,
                "file_path" : str(dest_path.relative_to(output_dir.parent)),
                "src_path"  : str(src_path),
            })
        except Exception as e:
            log.warning(f"  Lỗi khi xử lý {src_path.name}: {e}")

    log.info(f"[Lưu] Đã ghi {len(records)} file vào {output_dir}")
    return records


def print_summary(records: list[dict]) -> None:
    """In báo cáo phân bổ file theo speaker sau khi xử lý xong."""
    from collections import Counter
    counts = Counter(r["speaker"] for r in records)

    log.info("=" * 55)
    log.info(f"  TỔNG KẾT TIỀN XỬ LÝ")
    log.info("=" * 55)
    log.info(f"  Tổng file đã xử lý : {len(records)}")
    log.info(f"  Số speakers        : {len(counts)}")
    log.info(f"  File/speaker (min) : {min(counts.values())}")
    log.info(f"  File/speaker (max) : {max(counts.values())}")
    log.info(f"  File/speaker (avg) : {len(records)/len(counts):.1f}")
    log.info("=" * 55)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> list[dict]:
    """
    Chạy toàn bộ pipeline tiền xử lý từ đầu đến cuối.
    Trả về list metadata record – có thể dùng để insert DB ở bước tiếp theo.
    """
    log.info("━" * 55)
    log.info("  PIPELINE TIỀN XỬ LÝ VCTK CORPUS – BẮT ĐẦU")
    log.info("━" * 55)

    # Kiểm tra đường dẫn đầu vào
    if not SPEAKER_INFO.exists():
        raise FileNotFoundError(f"Không tìm thấy: {SPEAKER_INFO}")
    if not WAV48_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy: {WAV48_DIR}")

    # ── Bước 1 ────────────────────────────────────────────────────────────────
    female_speakers = load_female_speakers(SPEAKER_INFO)

    # ── Bước 2 ────────────────────────────────────────────────────────────────
    files_by_speaker = collect_wav_files(WAV48_DIR, female_speakers)

    # ── Bước 3 ────────────────────────────────────────────────────────────────
    valid_files = filter_valid_files(files_by_speaker)

    # ── Bước 4+5 ──────────────────────────────────────────────────────────────
    selected = sample_files(valid_files)

    # ── Lưu file ──────────────────────────────────────────────────────────────
    records = save_processed_files(selected, PROCESSED_DIR)

    # ── Báo cáo ───────────────────────────────────────────────────────────────
    print_summary(records)

    log.info("  PIPELINE HOÀN TẤT ✓")
    log.info("━" * 55)

    return records


if __name__ == "__main__":
    run_pipeline()
