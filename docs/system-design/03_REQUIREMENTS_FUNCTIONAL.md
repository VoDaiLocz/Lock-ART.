# 03. Functional Requirements

## FR-01 Protect Single Image

- Input: một ảnh.
- Output: ảnh đã bảo vệ + tùy chọn JSON report.

## FR-02 Batch Protect

- Input: thư mục ảnh (recursive tùy chọn).
- Output: thư mục đích + thống kê tổng hợp.

## FR-03 Analyze Pair

- Input: original/protected.
- Output: PSNR/SSIM/protection score.

## FR-04 Profile Benchmark

- Chạy nhiều profile trên cùng input.
- Xuất report so sánh định dạng máy đọc được.

## FR-05 Collective Subject Mode

- Hỗ trợ split theo set (`set_A`, `set_B`) cho benchmark tập thể.
