# 🔬 AuraLock - Research Roadmap (Cập nhật)

> Tài liệu này mô tả lộ trình nghiên cứu thực tế cho AuraLock sau mốc `v0.1.0`.

---

## 1) Mục tiêu nghiên cứu

Xây dựng và đánh giá phương pháp bảo vệ artwork theo hướng:
1. Ảnh đầu ra vẫn dùng được cho con người
2. Giảm khả năng mô hình học/preserve style gốc
3. Có benchmark minh bạch, lặp lại được
4. So sánh profile có kiểm soát thay vì đánh giá cảm tính

---

## 2) Trọng tâm kỹ thuật

### A. Objective và proxy metric
- Cải thiện objective cho `collective` mode
- Tách rõ score theo từng thành phần (embedding/style/robust transforms)
- Theo dõi trade-off quality/protection theo profile

### B. Benchmark thực nghiệm
- Tăng số tập dữ liệu thử nghiệm (đại diện nhiều phong cách)
- Chuẩn hóa quy trình baseline vs protected
- Tạo mẫu báo cáo để so sánh giữa các lần chạy

### C. Robustness và reproducibility
- Đánh giá dưới các biến đổi thường gặp (resize, blur, compression)
- Chuẩn hóa metadata run để dễ truy vết
- Kiểm soát seed và môi trường runtime trong benchmark

---

## 3) Kế hoạch theo giai đoạn

### Giai đoạn 1 (ngắn hạn: 2-4 tuần)
- Hoàn thiện logging/metadata cho các lệnh chính
- Chuẩn hóa report JSON để tiện tổng hợp
- Rà soát lại preset profile theo benchmark hiện tại

### Giai đoạn 2 (trung hạn: 1-2 tháng)
- Mở rộng benchmark trên nhiều subject set
- Tự động hóa so sánh profile theo batch
- Tăng độ tin cậy CI cho các luồng benchmark khô (dry-run)

### Giai đoạn 3 (dài hạn: 2-3 tháng)
- Chạy benchmark thực tế trên GPU (LoRA/DreamBooth)
- Tổng hợp kết quả có kiểm định cơ bản
- Đề xuất hướng cải tiến objective dựa trên dữ liệu thực nghiệm

---

## 4) Năng lực hạ tầng khuyến nghị

- **CPU workflow**: đủ cho protect/analyze/batch cơ bản và benchmark dry-run
- **GPU workflow**: cần cho benchmark huấn luyện thực tế LoRA/DreamBooth
- **Container workflow**: ưu tiên Docker benchmark runtime để giảm sai lệch môi trường

---

## 5) Tiêu chí hoàn thành mỗi vòng nghiên cứu

Mỗi vòng lặp nên có:
- Bộ dữ liệu đầu vào xác định rõ
- Cấu hình profile/params được ghi lại
- Report JSON đầy đủ
- So sánh quality (`PSNR`, `SSIM`) và protection proxy
- Kết luận ngắn gọn + hành động tiếp theo

---

## 6) Tài liệu liên quan

- `README.md`: hướng dẫn sử dụng và snapshot kết quả hiện tại
- `docs/PRODUCT_AUDIT.md`: bối cảnh cải tiến sản phẩm và các gap còn lại
- `notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb`: notebook benchmark trên Colab/GPU
