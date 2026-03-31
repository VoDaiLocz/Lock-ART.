# 14. Performance and Scaling

## CPU Optimizations

- Tái sử dụng tensor tạm nếu có thể.
- Giảm chuyển đổi qua lại PIL/Numpy/Torch.
- Tối ưu kích thước làm việc (`working-size`).

## GPU Path

- Cho workload benchmark nặng.
- Nên cho phép mixed precision có kiểm soát.

## Batch Scaling

- Chunking theo thư mục lớn.
- Resume từ danh sách chưa xử lý.
