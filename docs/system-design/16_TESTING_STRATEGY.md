# 16. Testing Strategy

## Test Pyramid

- Unit tests: core metrics, profile config.
- Integration tests: pipeline + service orchestration.
- Smoke tests: CLI commands phổ biến.

## Data Fixtures

- Ảnh synthetic nhỏ để chạy nhanh.
- Fixture tách riêng cho benchmark nặng.

## CI Gates (đề xuất)

- `ruff check src`
- `black --check src`
- `pytest` (core)
- optional nightly benchmark regression.
