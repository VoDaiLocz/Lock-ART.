# 09. Experiment Workflows

## Local Study Loop

1. Chọn 1 ảnh chuẩn.
2. Chạy `benchmark` với 3-6 profile.
3. So PSNR/SSIM/protection score.
4. Chọn profile ứng viên.
5. Chạy batch trên tập nhỏ.
6. Lưu artifact + report vào thư mục versioned.

## Subject-set Loop

- Dùng split `set_A` để tune.
- Dùng `set_B` để kiểm thử holdout.
- Tránh tune trên tập đánh giá cuối.
