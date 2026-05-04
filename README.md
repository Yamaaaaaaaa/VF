# Hệ thống Tìm kiếm Âm thanh tương đồng (Voice Similarity Search)

Dự án này thực hiện hệ thống tìm kiếm âm thanh dựa trên đặc trưng giọng nói (Voice Similarity) sử dụng tập dữ liệu VCTK Corpus, PostgreSQL với pgvector và giao diện Web (React + FastAPI).

## 📋 Mục lục
1. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
2. [Giai đoạn 1: Xử lý dữ liệu](#giai-đoạn-1-xử-lý-dữ-liệu)
3. [Giai đoạn 2: Chạy ứng dụng Web](#giai-đoạn-2-chạy-ứng-dụng-web)
4. [Cấu trúc dự án](#cấu-trúc-dự-án)

---

## 💻 Yêu cầu hệ thống
- **Python**: 3.10 trở lên.
- **Docker & Docker Compose**: Để chạy Database và Web services.
- **Dữ liệu**: VCTK Corpus đặt tại `data/raw/VCTK-Corpus/VCTK-Corpus/`.

---

## ⚙️ Giai đoạn 1: Xử lý dữ liệu (Data Processing)

Quy trình này biến đổi dữ liệu âm thanh thô thành các vector đặc trưng có thể tìm kiếm được.

### 1.1 Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```

### 1.2 Khởi tạo Cơ sở dữ liệu (Database)
Bạn có thể khởi động riêng Database để phục vụ việc xử lý dữ liệu:
```bash
docker compose up -d db
```
*Lưu ý: Bảng `voice_records` sẽ được tự động khởi tạo qua `init.sql`.*

### 1.3 Tiền xử lý âm thanh
Thực hiện resample, chuẩn hóa độ dài và lọc giọng nữ:
```bash
# Đứng tại thư mục gốc dự án (Beta/)
python src/preprocessing/preprocess.py
```
Kết quả lưu tại: `data/processed/`.

### 1.4 Trích xuất đặc trưng & Nạp dữ liệu (Ingest)
Thực hiện qua các Jupyter Notebook để dễ dàng theo dõi:
1. **Trích xuất**: Chạy `src/feature_extraction/feature_extraction.ipynb` để tạo vector embedding 99 chiều.
2. **Nạp dữ liệu**: Chạy `src/database/ingest.ipynb` để đẩy toàn bộ metadata và vector vào database.

---

## 🚀 Giai đoạn 2: Chạy ứng dụng Web (Web Application)

### 2.1 Khởi động toàn bộ dịch vụ (DB, Backend & Frontend)
Bây giờ bạn có thể khởi động tất cả các thành phần chỉ với **một lệnh duy nhất** từ thư mục gốc:
```bash
docker compose up --build -d
```

### 2.2 Truy cập giao diện
- **Frontend**: [http://localhost:5173](http://localhost:5173) (Giao diện tìm kiếm)
- **Backend API**: [http://localhost:8000/docs](http://localhost:8000/docs) (Tài liệu API Swagger)

---

## 📂 Cấu trúc dự án chính
- `src/preprocessing/`: Mã nguồn tiền xử lý âm thanh.
- `src/feature_extraction/`: Trích xuất MFCC, Spectral Contrast, Chroma.
- `src/database/`: Cấu hình Docker DB và script nạp dữ liệu.
- `src/web/backend/`: API FastAPI xử lý tìm kiếm và truy vấn vector.
- `src/web/frontend/`: Giao diện React hiển thị kết quả và biểu đồ.
- `data/`: Chứa dữ liệu thô, dữ liệu đã xử lý và các file đặc trưng.

---

## 💡 Lưu ý
- Nếu bạn chạy Backend không qua Docker, hãy đảm bảo biến môi trường `DB_HOST` trong code trỏ về `localhost` thay vì `host.docker.internal`.
- Quá trình trích xuất đặc trưng cho 500 file mất khoảng 1-2 phút tùy cấu hình máy.
