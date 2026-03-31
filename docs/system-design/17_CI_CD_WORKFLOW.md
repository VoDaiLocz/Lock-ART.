# 17. CI/CD Workflow

## Pull Request Flow

1. Lint + unit test bắt buộc.
2. Kiểm tra thay đổi docs nếu touch CLI/API.
3. Comment tự động nếu thiếu benchmark evidence cho thay đổi profile.

## Release Flow

- Tag semantic version.
- Xuất changelog tóm tắt thay đổi + giới hạn.
- Đính kèm báo cáo benchmark chuẩn.
