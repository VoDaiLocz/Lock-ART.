# 10. Benchmark Design

## Objective

Đo tương quan giữa độ bền anti-mimicry proxy và chất lượng ảnh.

## Dimensions

- Image quality: PSNR, SSIM.
- Protection proxy: embedding/style drift sau biến đổi robust.
- Compute cost: thời gian/ảnh, RAM.

## Protocol

- Cố định seed khi có thể.
- Chuẩn hóa resize/crop trước khi so sánh.
- Báo cáo trung bình + p95 trên batch.

## Anti-bias

- Không cherry-pick ảnh đẹp nhất.
- Tách tập tune/test.
