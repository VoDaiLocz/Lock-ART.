# 00. System Design Overview

Tài liệu này mô tả thiết kế hệ thống cho AuraLock theo hướng **nghiên cứu có thể lặp lại**,
ưu tiên tính minh bạch, khả năng benchmark và mở rộng an toàn.

## Mục tiêu

- Chuẩn hóa kiến trúc end-to-end từ CLI/UI đến core pipeline.
- Mô tả dataflow cho bảo vệ ảnh đơn, batch và benchmark.
- Định nghĩa phi chức năng: hiệu năng, quan sát, bảo mật, độ tin cậy.
- Làm nền cho cộng tác kỹ thuật và roadmap dài hạn.

## Đối tượng

- Contributor kỹ thuật (core/ML/infra).
- Người dùng nghiên cứu cần tái lập kết quả.
- Maintainer phụ trách release và chất lượng.

## Cách đọc

- Bắt đầu từ `01_ARCHITECTURE_CONTEXT.md` đến `05_COMPONENTS.md`.
- Sau đó đọc các mảng: dữ liệu, bảo mật, quan sát, CI/CD.
- Cuối cùng đọc roadmap và risk register để hiểu trade-off hiện tại.
