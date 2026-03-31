# 13. Error Handling Strategy

## Error Classes (đề xuất)

- `InputValidationError`
- `ProfileConfigError`
- `PipelineExecutionError`
- `ReportSerializationError`

## Policy

- CLI trả exit code khác nhau theo nhóm lỗi.
- Batch không dừng toàn bộ khi một file lỗi (trừ lỗi hệ thống nghiêm trọng).
- Report ghi danh sách file thất bại để retry.
