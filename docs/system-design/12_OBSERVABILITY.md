# 12. Observability and Diagnostics

## Logging

- Mức `INFO`: tiến trình theo file/profile.
- Mức `DEBUG`: tham số chi tiết và transform chain.
- Mức `ERROR`: lỗi có mã + context.

## Metrics to track

- thời gian xử lý/ảnh,
- tỷ lệ lỗi batch,
- phân phối protection score,
- phân phối PSNR/SSIM.

## Artifact Hygiene

- Tên file có timestamp + run_id.
- Report JSON phải có version schema.
