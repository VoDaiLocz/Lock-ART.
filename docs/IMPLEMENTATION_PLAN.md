# 🛡️ AuraLock - Kế Hoạch Triển Khai (Cập nhật)

> **Repository**: `VoDaiLocz/Lock-ART.`  
> **Phiên bản hiện tại**: `0.1.0`  
> **Mục tiêu**: tài liệu hóa trạng thái triển khai thực tế và hướng phát triển ngắn hạn.

---

## 1) Tổng quan

AuraLock là toolkit phục vụ **nghiên cứu bảo vệ artwork** trước nguy cơ bị mô hình sinh ảnh học và bắt chước phong cách.

Nguyên tắc của dự án:
- Ưu tiên **thử nghiệm có thể lặp lại** (reproducible)
- Đo lường bằng **chỉ số minh bạch** (PSNR, SSIM, protection proxy)
- Không đưa ra tuyên bố tuyệt đối kiểu “chặn mọi AI”

---

## 2) Trạng thái triển khai hiện tại

### ✅ Core và protection workflow
- Image I/O và metrics nền tảng
- FGSM/PGD và pipeline bảo vệ theo profile
- Các profile sử dụng nhanh: `safe`, `balanced`, `strong`, `subject`, `fortress`, `blindfold`
- Chế độ adaptive guardrails cho CLI `protect`

### ✅ CLI, batch và benchmark
- CLI đầy đủ cho `protect`, `analyze`, `batch`, `benchmark`
- Batch theo thư mục và chế độ `--collective` cho subject set
- Xuất report JSON cho workflow tự động

### ✅ UI và đóng gói
- Web UI (Gradio) ở dạng tùy chọn (`.[ui]`)
- Entry point CLI và webui
- Dockerfile cho runtime cơ bản và benchmark runtime

### ✅ Kiểm thử và CI
- Test suite bằng `pytest`
- Lint/format bằng `ruff` và `black`
- CI workflow trên GitHub Actions

---

## 3) Mục tiêu ngắn hạn (roadmap thực thi)

### Ưu tiên cao
- Bổ sung logging có cấu trúc cho từng run
- Tăng chất lượng tổng hợp report đa-run
- Xác thực benchmark runtime trên GPU thực tế

### Ưu tiên trung bình
- Củng cố benchmark dataset và báo cáo chuẩn hóa
- Tăng end-to-end coverage cho các luồng CLI chính
- Chuẩn hóa release/changelog

### Ưu tiên nghiên cứu
- Đánh giá sâu hơn với các pipeline LoRA / DreamBooth thực tế
- Mở rộng robustness checks với các quy trình mimicry phổ biến

---

## 4) Hướng dẫn vận hành nhanh

### Cài đặt phát triển
```bash
git clone https://github.com/VoDaiLocz/Lock-ART.
cd Lock-ART.
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

### Lệnh thường dùng
```bash
auralock protect artwork.png -o protected.png --profile balanced
auralock analyze original.png protected.png --report reports/analyze.json
auralock batch ./artworks ./protected --recursive
auralock benchmark artwork.png --profiles safe,balanced,strong --report reports/benchmark.json
```

### Chạy kiểm tra chất lượng
```bash
pytest -q
ruff check src
black --check src
```

---

## 5) Cấu trúc dự án (rút gọn)

```text
Lock-ART./
+-- docs/
+-- notebooks/
+-- src/
|   +-- auralock/
|   |   +-- attacks/
|   |   +-- benchmarks/
|   |   +-- core/
|   |   +-- services/
|   |   +-- ui/
|   |   \-- cli.py
|   \-- tests/
+-- .github/workflows/
+-- Dockerfile
+-- Dockerfile.benchmark
+-- pyproject.toml
\-- README.md
```

---

## 6) Ghi chú

Tài liệu này mô tả trạng thái triển khai theo hướng sản phẩm nghiên cứu. Để xem hướng dẫn sử dụng đầy đủ và snapshot benchmark hiện tại, ưu tiên tham chiếu `README.md`.
