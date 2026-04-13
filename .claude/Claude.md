# Claude.md – Hệ thống Tìm kiếm Giọng nói Tương đồng (Voice Similarity Search)

---

## P1: Tổng quan dự án

- **Mô tả:** Hệ thống tìm kiếm và so khớp giọng nói dựa trên đặc trưng âm thanh, sử dụng vector embedding để tìm top-5 giọng nói tương đồng nhất từ cơ sở dữ liệu.
- **Mục tiêu:** Cho phép người dùng upload một file `.wav`, hệ thống trích xuất đặc trưng, tra cứu trong CSDL và trả về các giọng nói giống nhất kèm thông tin metadata và công cụ nghe/so sánh.
- **Tính năng chính:**
  - Upload & phát lại file `.wav` đầu vào, hiển thị dạng sóng.
  - Trích xuất đặc trưng âm thanh (MFCC, Spectral Contrast, Chroma STFT).
  - Tìm kiếm vector tương đồng trong PostgreSQL + pgvector.
  - Hiển thị top-5 kết quả với rank, độ tương đồng, metadata, nút play và nút chi tiết.
  - Biểu đồ so sánh đặc trưng để đánh giá sự tương đồng.

---

## P2: Tech Stack

- **Frontend:** Web (HTML/CSS/JS hoặc framework tùy chọn)
- **Backend:** Python (xử lý âm thanh, trích xuất đặc trưng, API)
- **Database:**
  - PostgreSQL + pgvector (lưu trữ metadata và vector embedding)
  - File System (lưu file `.wav` gốc và file `.npy` đặc trưng thô)
- **Thư viện âm thanh:** librosa, numpy, scipy
- **Dataset:** CSTR VCTK Corpus

---

## P3: Thu thập & Tiền xử lý Dữ liệu

> **Tài liệu chi tiết:** `documents/preprocess_pipeline.md`  
> **Code:** `src/preprocessing/preprocess.ipynb`

### 3.1 Tổng quan Dataset

| Thuộc tính | Giá trị |
|---|---|
| **Nguồn** | CSTR VCTK Corpus v0.80 |
| **Tổng speakers** | 109 người (p225 – p376) |
| **Số câu / speaker** | ~400 câu/người |
| **Sample rate gốc** | **48 kHz**, 16-bit (thư mục `wav48/`) |
| **Metadata** | `speaker-info.txt` → ID, AGE, GENDER, ACCENTS, REGION |
| **Số speakers nữ (F)** | ~65 speakers → đủ lấy 500 file |

---
### 3.2 Yêu cầu 
1. Cắt/pad file âm thanh về đúng 5 giây.
2. Resample về 16.000 Hz nếu chưa đúng.
3. Lọc theo trường `gender = female` trong metadata VCTK.
4. Chỉ lấy đúng 500 file
5. Giới tính: **female** (chỉ lấy giọng nữ)
### 3.3 Cấu trúc nguồn dữ liệu thực tế

```
data/raw/VCTK-Corpus/VCTK-Corpus/
├── speaker-info.txt     # Metadata: ID AGE GENDER ACCENTS REGION
├── wav48/               # Audio gốc 48kHz
│   ├── p225/            # p225 = female, 23 tuổi, English
│   │   ├── p225_001.wav
│   │   ├── p225_002.wav
│   │   └── ...          # ~400 file/speaker
│   ├── p226/            # p226 = male → sẽ bị lọc bỏ
│   └── ...
└── txt/                 # Transcript (không dùng trong project này)
```

> **Lưu ý quan trọng:** File gốc là **48 kHz**, không phải 16 kHz → **bắt buộc phải resample** trước khi trích xuất đặc trưng.

---

### 3.4 Pipeline Tiền xử lý (5 bước)

```
[speaker-info.txt]
       │
  [Bước 1] Đọc & lọc metadata
  → Chỉ giữ speakers có GENDER = F (~65 speakers)
       │
  [Bước 2] Duyệt file .wav của các female speakers
  → Lấy tất cả file từ wav48/<speaker>/
       │
  [Bước 3] Kiểm tra chất lượng file
  → Bỏ file bị lỗi, silence hoàn toàn, hoặc quá ngắn (< 1s)
       │
  [Bước 4] Resample 48kHz → 16kHz  +  Cắt/Pad → 5 giây cố định
       │
  [Bước 5] Sampling 500 file (phân bổ đều theo speaker)
  → Lưu vào data/processed/<speaker>/<file>.wav
```

---

### 3.5 Chi tiết từng bước

#### Bước 1 – Đọc & lọc metadata từ `speaker-info.txt`

```python
import pandas as pd

def load_female_speakers(speaker_info_path: str) -> list[str]:
    """
    Đọc speaker-info.txt, trả về list speaker_id của giọng nữ.
    Định dạng: ID  AGE  GENDER  ACCENTS  REGION
    """
    speakers = []
    with open(speaker_info_path, 'r') as f:
        next(f)  # Bỏ header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                speaker_id = parts[0]   # VD: "225"
                gender     = parts[2]   # VD: "F" hoặc "M"
                if gender == 'F':
                    speakers.append(f"p{speaker_id}")  # → "p225"
    return speakers

# Kết quả: ['p225', 'p228', 'p229', 'p230', 'p231', ...] (~65 speakers)
```

#### Bước 2 – Duyệt file .wav của female speakers

```python
from pathlib import Path

def collect_wav_files(wav48_dir: str, female_speakers: list[str]) -> list[Path]:
    wav48 = Path(wav48_dir)
    all_files = []
    for speaker in female_speakers:
        speaker_dir = wav48 / speaker
        if speaker_dir.exists():
            files = sorted(speaker_dir.glob("*.wav"))
            all_files.extend(files)
    return all_files
```

#### Bước 3 – Kiểm tra chất lượng file

| Điều kiện loại bỏ | Lý do |
|---|---|
| File không đọc được (corrupt) | Librosa raise exception |
| Duration < 1 giây | Quá ngắn, không đủ thông tin đặc trưng |
| RMS năng lượng ≈ 0 (silence) | File trống / thu lỗi |

```python
import librosa, numpy as np

def is_valid_audio(wav_path: str, min_duration: float = 1.0) -> bool:
    try:
        y, sr = librosa.load(wav_path, sr=None)  # Giữ sr gốc
        duration = len(y) / sr
        rms = np.sqrt(np.mean(y**2))
        return duration >= min_duration and rms > 1e-4
    except Exception:
        return False
```

#### Bước 4 – Resample 48kHz → 16kHz & Cắt/Pad về 5 giây

```python
TARGET_SR       = 16_000   # Hz
TARGET_DURATION = 5.0      # giây
TARGET_SAMPLES  = int(TARGET_SR * TARGET_DURATION)  # = 80,000 mẫu

def preprocess_audio(wav_path: str) -> np.ndarray:
    """
    Load → resample → cắt hoặc pad về đúng 5 giây @ 16kHz.
    Trả về numpy array shape (80000,).
    """
    # Load + resample (librosa tự xử lý 48kHz → 16kHz)
    y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    if len(y) >= TARGET_SAMPLES:
        # Cắt lấy 5 giây đầu
        y = y[:TARGET_SAMPLES]
    else:
        # Pad zero ở cuối
        pad_len = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')

    return y   # shape: (80000,)
```

> **Tại sao cắt từ đầu chứ không cắt giữa?**  
> Câu trong VCTK thường có nội dung quan trọng ở phần đầu, phần cuối hay bị silence. Cắt đầu cho kết quả ổn định hơn.

#### Bước 5 – Sampling 500 file (phân bổ đều)

```python
import random

def sample_files(valid_files_by_speaker: dict[str, list[Path]],
                 total: int = 500,
                 seed: int = 42) -> list[Path]:
    """
    Phân bổ đều: mỗi speaker lấy khoảng total // n_speakers file.
    Nếu speaker không đủ → lấy hết, speaker khác bù vào.
    """
    random.seed(seed)
    n_speakers = len(valid_files_by_speaker)
    quota = total // n_speakers  # ~7-8 file/speaker

    selected = []
    leftover_pool = []
    for files in valid_files_by_speaker.values():
        shuffled = random.sample(files, len(files))
        selected.extend(shuffled[:quota])
        leftover_pool.extend(shuffled[quota:])

    # Bù nếu chưa đủ 500
    deficit = total - len(selected)
    if deficit > 0:
        selected.extend(random.sample(leftover_pool, deficit))

    return selected[:total]
```

> **Tại sao phân bổ đều theo speaker?**  
> Tránh bias: nếu lấy ngẫu nhiên hoàn toàn, một speaker nhiều file sẽ chiếm đa số → vector embedding bị kéo về không gian của speaker đó.

---

### 3.6 Tổng kết bộ lọc áp dụng

| Bộ lọc | Giá trị | Lý do |
|---|---|---|
| **GENDER** | `F` (female) | Thu hẹp không gian giọng nói, tăng độ tương đồng nội bộ |
| **Duration** | ≥ 1s (hợp lệ), cắt/pad → **5s** | Đồng nhất input cho trích xuất đặc trưng |
| **Sample Rate** | Resample → **16 kHz** | Chuẩn ASR/TTS, giảm tính toán so với 48kHz |
| **Chất lượng** | Loại silence & file lỗi | Tránh vector nhiễu trong DB |
| **Số lượng** | **500 file**, phân bổ đều | Đủ để demo, không quá lớn cho pgvector IVFFlat |

---

## P4: Trích xuất Đặc trưng

> **Tài liệu chi tiết:** `documents/feature_extraction.md`  
> **Code:** `src/feature_extraction/feature_extraction.ipynb`

### 4.1 Tổng quan Vector 99 Chiều

```
[0:40]   MFCC mean          (40 chiều) – hình dạng phổ Mel trung bình
[40:80]  MFCC std           (40 chiều) – độ biến động giọng nói
[80:87]  Spectral Contrast  ( 7 chiều) – tương phản phổ tần số
[87:99]  Chroma STFT        (12 chiều) – phân bố pitch class
──────────────────────────────────────
TOTAL                       99 chiều  → VECTOR(99) trong pgvector
```

### 4.2 Bảng Tham Số Chi Tiết

| Đặc trưng | Chiều | Thư viện | Tham số chính | Nắm bắt |
|---|---|---|---|---|
| **MFCC mean** | 40 | `librosa.feature.mfcc` | `n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128` | Màu sắc âm thanh trung bình |
| **MFCC std** | 40 | `librosa.feature.mfcc` | (same) | Độ biến thiên nhịp điệu giọng |
| **Spectral Contrast mean** | 7 | `librosa.feature.spectral_contrast` | `n_bands=6, fmin=200` | Tương phản peak/valley 6 dải tần |
| **Chroma STFT mean** | 12 | `librosa.feature.chroma_stft` | `n_fft=2048, hop_length=512` | Phân bố 12 pitch class (C–B) |
| **Tổng** | **99** | | | |

### 4.3 Tham Số Cấu Hình

```python
TARGET_SR  = 16_000   # Hz
N_MFCC     = 40       # Số hệ số MFCC (chuẩn speaker recognition)
N_FFT      = 2048     # Cửa sổ FFT = 128ms @ 16kHz
HOP_LENGTH = 512      # Bước nhảy = 32ms, overlap 75%
N_MELS     = 128      # Mel filter banks
SC_N_BANDS = 6        # Spectral Contrast sub-bands → 7 hệ số
SC_FMIN    = 200.0    # Hz – bắt đầu dải Spectral Contrast
EMBEDDING_DIM = 99    # = 40+40+7+12
```

### 4.4 Hàm Trích Xuất

```python
def extract_features(wav_path: Path) -> dict:
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

    # MFCC: (40, 157) → mean (40,) + std (40,)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    mfcc_std  = np.std(mfcc,  axis=1).astype(np.float32)

    # Spectral Contrast: (7, 157) → mean (7,)
    sc = librosa.feature.spectral_contrast(y=y, sr=sr,
                                            n_bands=SC_N_BANDS, fmin=SC_FMIN)
    sc_mean = np.mean(sc, axis=1).astype(np.float32)

    # Chroma STFT: (12, 157) → mean (12,)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)

    return {"mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std,
            "sc_mean": sc_mean, "chroma_mean": chroma_mean,
            "mfcc_raw": mfcc}  # (40,157) → lưu .npy cho DTW


def build_embedding(features: dict) -> np.ndarray:
    return np.concatenate([features["mfcc_mean"], features["mfcc_std"],
                           features["sc_mean"], features["chroma_mean"]])
    # shape: (99,) float32
```

### 4.5 Output

| Output | Vị trí | Format | Mục đích |
|---|---|---|---|
| **Vector embedding** | PostgreSQL `VECTOR(99)` | `list[float]` → pgvector | Tìm kiếm KNN nhanh |
| **MFCC raw** | `data/features/<spk>/<file>.npy` | `(40,157) float32` | DTW, phân tích chuỗi thời gian |
| **Records JSON** | `data/feature_records.json` | JSON backup | Insert DB, debug |

### 4.6 Lý Giải Lựa Chọn Đặc Trưng

- **MFCC:** Mô phỏng thính giác người, bất biến với cường độ âm, phân biệt formant
- **Spectral Contrast:** Bắt được tính "rõ ràng / mờ nhạt" của phổ – bổ sung cho MFCC
- **Chroma:** Phân bố pitch class – phân biệt giọng cao/trầm, robust với octave shift
- **Không dùng:** Zero Crossing Rate, RMS (quá đơn giản, ít phân biệt tính giọng)

---

## P5: Hệ Cơ sở Dữ liệu

### PostgreSQL + pgvector
- **Bảng chính:** Lưu metadata + embedding
- **Cấu trúc cột:**

```sql
CREATE TABLE voice_records (
    file_id      SERIAL PRIMARY KEY,
    speaker      TEXT,           -- ID/tên người nói (VD: p225)
    accent       TEXT,           -- Giọng vùng miền
    gender       TEXT,           -- 'female' / 'male'
    file_path    TEXT,           -- Đường dẫn đến file .wav
    npy_path     TEXT,           -- Đường dẫn đến file .npy đặc trưng thô
    embedding    VECTOR(N)       -- Vector tổng hợp các đặc trưng (N = số chiều)
);
```

- **Tìm kiếm:** Dùng `pgvector` với toán tử `<=>` (cosine distance) hoặc `<->` (L2 distance).

### Lý do chọn pgvector thay vì Cây Thuật Toán Truyền Thống (Quadtree, R-Tree, K-D Tree)

Trong khuôn khổ hệ CSDL Đa phương tiện, thay vì tự cấu trúc các Cây Không Gian (Spatial Tree) bằng tay, dự án quyết định sử dụng `pgvector` kết hợp index `ivfflat`. Nguyên nhân cốt lõi bao gồm:

1. **Lời nguyền Không gian Đa chiều (Curse of Dimensionality):**
   Đặc trưng vector của dự án sinh ra tới **99 chiều (99-D)**. Ở không gian khổng lồ này, các cấu trúc Cây kinh điển sẽ hoàn toàn sụp đổ:
   - **Quadtree / MX-Quadtree:** Tại mỗi Node phải chia $2^d$ nhánh con. Với $d=99$, số nhánh cấp 1 sinh ra là $2^{99}$ (bất khả thi về mặt vật lý).
   - **R-Tree / R*-Tree:** Hoạt động tốt nhất ở $<10$ chiều. Ở $99D$, các khối Bounding Box bao quanh vector gần như đè chồng lên nhau hoàn toàn (Overlap > 95%). Thuật toán tìm kiếm sẽ phải đi vào toàn bộ các nhánh con, khiến hiệu năng truy vấn suy thoái thành $O(N)$ (chậm như duyệt mảng thông thường).
   - **K-D Tree:** Tốc độ tìm kiếm ở 99D sẽ rơi tiệm cận $O(N)$, hoàn toàn làm mất đi lợi thế cây tìm kiếm nhị phân $O(\log N)$.

2. **Tiêu chuẩn Thực tiễn thế hệ mới (State-of-The-Art):**
   Thuật toán **ANN (Approximate Nearest Neighbor)** như `ivfflat` (mà pgvector đang dùng) là chuẩn mực để giải quyết "Tìm kiếm Vector đa chiều" của AI. Việc áp dụng đúng kỹ thuật này thể hiện tính thực tiễn cao cho đồ án so với việc bắt ép các thuật toán cũ làm sai chức năng.

### Triển khai CSDL & Đẩy dữ liệu (Ingestion Pipeline)

Toàn bộ script cấu hình hệ thống lưu trữ nằm trong thư mục `src/database/`:
1. **Khởi tạo Docker:** Cấu hình `docker-compose.yml` (chạy image `pgvector/pgvector:pg16`). Khi build, kịch bản `init.sql` sẽ tự động enable extension `vector`, xây dựng schema `voice_records` và tạo index `ivfflat` giúp tìm kiếm ANN.
2. **Ingest Script (`ingest.ipynb`):** Notebook này thực thi thao tác gộp dữ liệu bằng Python (`psycopg2`), bao gồm:
   - Tải thông tin người nói (Age, Gender, Accent) từ `raw/VCTK-Corpus/VCTK-Corpus/speaker-info.txt`.
   - Kết nối với Vector Embedding (99 chiều) từ `feature_records.json`.
   - Đối soát file `.wav` từ thư mục `data/processed` để chắc chắn file tồn tại.
   - Insert tự động vào hệ quản trị CSDL với chuẩn VECTOR data type.

### File System
```
data/
├── raw/          # File .wav gốc chưa xử lý (từ VCTK Corpus)
├── processed/    # File .wav đã qua tiền xử lý (16kHz, 5s)
└── features/     # File .npy chứa đặc trưng thô theo thời gian
```

---

## P6: Lưu trữ (Metadata + Features Vector)

> **Tài liệu chi tiết chiến lược lưu trữ:** `documents/storage_strategy.md`

Chiến lược lưu trữ tối ưu của hệ thống hướng tới sự tách bạch rõ ràng giữa **Metadata** (lưu ở PostgreSQL và file `.csv`) và **Features Vector** (lưu ở pgvector và file array `.npy`). Việc này đảm bảo tối ưu truy vấn nhanh, không gây cồng kềnh database và dễ bề mở rộng. Chi tiết cấu trúc SQL và Ingest Pipeline tham khảo trong tài liệu lưu trữ.

---

## P7: Giao diện Web

### Input
- Nút **upload file `.wav`**.
- **Phát lại** file vừa upload (audio player).
- Hiển thị **waveform** (dạng sóng) của file đầu vào.

### Trung gian (Processing Display)
- Hiển thị các **biểu đồ đặc trưng** đã trích xuất (MFCC, Spectral Contrast, Chroma STFT).
- Bảng thống kê giá trị mean/std.

### Output – Top 5 Matches
Mỗi kết quả hiển thị:
| Trường | Mô tả |
|---|---|
| **Rank** | Thứ hạng (1–5) |
| **Độ tương đồng** | Cosine similarity / khoảng cách vector |
| **Metadata** | speaker, accent, gender, file_path |
| **Nút Play** | Nghe file kết quả để so sánh |
| **Nút Chi tiết** *(optional)* | Xem biểu đồ chồng lấp đặc trưng giữa query và kết quả |

---

## P8: Đánh giá Sự Tương đồng

> Tài liệu các phương án đánh giá dự kiến sẽ đặt trong `docs/evaluation/`.

### Mục tiêu
Chứng minh **tại sao** hai giọng nói được coi là tương đồng, ngoài việc dùng tai nghe.

### Phương án 1: Biểu đồ Tần số (Spectral Overlay)
- Vẽ **biểu đồ chồng lấp** (overlay) MFCC / Spectrogram của query và kết quả.
- Trực quan hóa sự biến đổi tần số theo thời gian.
- Dùng để thấy rằng hai giọng có hình dạng phổ tương tự.

### Phương án 2: So khớp Chuỗi Thời gian – DTW
- **Vấn đề:** Mean giống nhau không có nghĩa là phát âm giống nhau (cùng trung bình nhưng thứ tự khác).
- **Giải pháp:** Dùng **Dynamic Time Warping (DTW)** trên chuỗi MFCC theo frame.
- File `.npy` trong `data/features/` lưu đặc trưng thô theo thời gian phục vụ chính xác mục đích này.
- DTW distance nhỏ → chuỗi âm thanh tương đồng về cấu trúc thời gian.

### Phương án 3: Cosine Similarity (Primary Search Metric)
- Dùng cho **tìm kiếm nhanh** trong pgvector.
- Giá trị gần 1.0 → rất tương đồng.

### Tóm tắt Phương án Đánh giá
| Phương án | Ưu điểm | Hạn chế |
|---|---|---|
| Spectral Overlay | Trực quan, dễ giải thích | Không bắt được thứ tự thời gian |
| DTW | Chính xác về chuỗi thời gian | Tốn tài nguyên tính toán |
| Cosine Similarity | Nhanh, phù hợp tìm kiếm real-time | Mất thông tin thứ tự |

---

## P9: Quy tắc Bắt buộc

- Không hardcode đường dẫn file – dùng biến môi trường hoặc config.
- Không commit file `.wav`, `.npy` lên git – lưu trong `.gitignore`.
- Không để lộ thông tin kết nối database (host, password) trong code.
- Luôn validate file upload: chỉ chấp nhận `.wav`, kiểm tra sample rate và độ dài.
- Xử lý lỗi rõ ràng khi file không đủ điều kiện (quá ngắn, sai format, v.v.).

---

## P10: Cấu trúc Thư mục Đề xuất

```
Beta/
├── data/
│   ├── raw/              # File .wav chưa xử lý
│   ├── processed/        # File .wav đã tiền xử lý
│   ├── features/         # File .npy đặc trưng thô theo thời gian
│   └── metadata/         # File CSV lưu metadata âm thanh
├── src/
│   ├── preprocessing/    # Script tiền xử lý dữ liệu VCTK
│   ├── feature_extraction/ # Trích xuất MFCC, Spectral Contrast, Chroma STFT
│   ├── extract_metadata/ # Chạy script lấy metadata âm thanh ra CSV
│   ├── database/         # Kết nối PostgreSQL, insert/query pgvector
│   ├── api/              # Backend API endpoints
│   └── web/              # Frontend (upload, waveform, kết quả)
├── documents/
│   ├── audio_metadata.md     # Tài liệu hệ thống metadata (Acoustic/Non-acoustic)
│   ├── feature_extraction.md # Tài liệu trích xuất đặc trưng
│   ├── preprocess_pipeline.md# Tài liệu pipeline tiền xử lý
│   └── storage_strategy.md   # Tài liệu Cấu trúc API lưu trữ (Database & Vector)
├── docs/
│   ├── features/         # Tài liệu chi tiết + code mẫu trích xuất đặc trưng
│   └── evaluation/       # Tài liệu các phương án đánh giá
├── .claude/
│   ├── Claude.md         # File này
│   └── agents/
└── .gitignore
```



## P11: Chức năng làm thêm (Priority thấp)
- Tìm kiếm theo Metadata 