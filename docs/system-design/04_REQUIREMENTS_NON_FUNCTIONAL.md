# 04. Non-Functional Requirements

## Performance

- Ảnh 512x512 CPU phải hoàn tất trong thời gian thực dụng nghiên cứu.
- Batch phải có tiến trình rõ ràng để theo dõi.

## Reliability

- Report phải ghi lỗi có cấu trúc thay vì mất log.
- Chạy lặp cùng config cho sai lệch chấp nhận được.

## Usability

- CLI command nhất quán, gợi ý tham số rõ.
- README + docs đủ để người mới chạy trong <30 phút.

## Portability

- Hỗ trợ Python 3.10+.
- Chạy Windows/Linux/macOS cho luồng core.

## Maintainability

- Module boundaries rõ và test theo lớp.
