# 07. Data Model

## Core Entities

- `ImageInput`: đường dẫn, mode, kích thước gốc.
- `ProtectionConfig`: profile + override epsilon/steps/seed.
- `ProtectionResult`: output path, metrics, warnings.
- `BenchmarkResult`: nhiều `ProtectionResult` + ranking.

## Report Schema (đề xuất)

- `run_id`, `timestamp`, `profile`.
- `input_meta` (size, channels, dtype).
- `metrics` (`psnr`, `ssim`, `protection_score`).
- `transforms_robustness`.
- `environment` (python, torch, device).
- `notes` và `limitations`.
