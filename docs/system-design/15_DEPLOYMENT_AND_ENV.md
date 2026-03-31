# 15. Deployment & Environment

## Local Dev

- `pip install -e .[dev]`
- Chạy CLI trực tiếp.

## Docker

- `Dockerfile` cho runtime core.
- `Dockerfile.benchmark` cho workflow benchmark chuyên sâu.

## Reproducible Runs

- Pin version dependency khi xuất bản benchmark lớn.
- Lưu thông tin CUDA/driver nếu dùng GPU.
