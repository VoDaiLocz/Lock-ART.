# 08. Profile Strategy

## Existing Presets

- `safe`: ưu tiên fidelity.
- `balanced`: baseline mặc định khuyến nghị.
- `strong`: tăng drift.
- `subject`: cho subject-style benchmark.
- `fortress`: aggressive hơn.
- `blindfold`: cực mạnh, giảm fidelity nhiều.

## Governance cho profile mới

- Cần benchmark tối thiểu trên tập ảnh đại diện.
- Không merge nếu chỉ cải thiện một metric mà phá mạnh metric còn lại.
- Bắt buộc ghi rõ failure cases.
