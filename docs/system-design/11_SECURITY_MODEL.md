# 11. Security Model

## Threats

- Path traversal qua input/output path.
- Ghi đè file ngoài ý muốn khi batch.
- Poisoned images làm crash pipeline.
- Lộ thông tin hệ thống qua report/log.

## Controls

- Chuẩn hóa/validate đường dẫn trước khi ghi.
- Tạo output trong thư mục đích được kiểm soát.
- Giới hạn kích thước ảnh đầu vào hợp lý.
- Redact thông tin nhạy cảm trong log.

## Residual Risks

- Tool nghiên cứu chưa phải sandbox bảo mật cứng.
