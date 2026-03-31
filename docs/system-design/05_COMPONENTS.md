# 05. Component Decomposition

## `auralock.cli`

- Parse command và orchestrate service.

## `auralock.services.protection`

- Lớp use-case chính, nối input -> pipeline -> report.

## `auralock.core.*`

- `image`: thao tác ảnh cơ bản.
- `style`: style feature extraction.
- `pipeline`: logic bảo vệ + adaptive.
- `metrics`: PSNR/SSIM/protection proxy.
- `profiles`: preset tham số.

## `auralock.attacks.*`

- FGSM/PGD/StyleCloak components dạng pluggable.

## `auralock.benchmarks.*`

- Manifest và adapter cho luồng benchmark chuyên sâu.

## `auralock.ui.gradio_app`

- Giao diện demo cho thao tác nhanh.
