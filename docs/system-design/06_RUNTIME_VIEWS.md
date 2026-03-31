# 06. Runtime Views

## Sequence: `protect`

1. CLI nhận input/path/profile.
2. Service load profile + normalize tham số.
3. Pipeline chạy attack chain theo budget.
4. Metrics engine tính chất lượng và proxy score.
5. Ghi ảnh + JSON report.

## Sequence: `benchmark`

1. Nạp danh sách profile.
2. Lặp profile và gọi protect/analyze.
3. Tổng hợp kết quả thành bảng JSON.
4. Trả exit code theo tiêu chí ngưỡng (nếu có).
