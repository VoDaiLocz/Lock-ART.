# AuraLock Product Audit

## Snapshot

AuraLock started as a clean research/demo repo with working algorithms, but it was still missing several traits that make a project behave like a professional product. Compared with the public `MiroFish` repository, the main gaps were not just "more features"; they were consistency, deployment readiness, and operational discipline.

## What Was Improved

### 1. Protection pipeline is now consistent
- Added a differentiable ImageNet preprocessing adapter in [`src/auralock/core/pipeline.py`](../src/auralock/core/pipeline.py)
- Stopped CLI and UI from each building their own model/input pipeline
- Preserved original output resolution while still feeding the classifier the normalized `224x224` input it expects

### 2. Shared service layer for product behavior
- Added [`src/auralock/services/protection.py`](../src/auralock/services/protection.py)
- Centralized attack execution, prediction reporting, quality metrics, style-readability metrics, and output formatting inputs
- Reduced duplication between CLI and Gradio UI

### 3. Stability hardening
- Added validation for invalid `epsilon`, `alpha`, and `num_steps`
- Made the optional UI import lazy so `auralock.ui` no longer crashes when `gradio` is not installed
- Fixed editable packaging support by declaring `editables` in `pyproject.toml`

### 4. Product workflow readiness
- Added CI in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
- Added a dedicated `auralock-webui` entry point
- Added tests covering the new adapter/service flow
- Added batch processing plus CLI integration tests
- Added Docker runtime files for one-command local deployment

### 5. Protection logic now matches the actual product goal better
- Added [`src/auralock/attacks/stylecloak.py`](../src/auralock/attacks/stylecloak.py)
- Added [`src/auralock/core/style.py`](../src/auralock/core/style.py)
- Added [`src/auralock/core/profiles.py`](../src/auralock/core/profiles.py)
- Moved the default workflow away from plain classifier fooling and toward feature/style-space cloaking with robust transforms
- Added a protection report that measures embedding and style similarity after blur and resize-restore transforms
- Added named protection profiles plus JSON report export for CLI-driven workflows
- Added stronger `balanced`/`subject` presets, a new `fortress` profile, a new aggressive `blindfold` obfuscation mode for anti-readability use cases, and adaptive CLI guardrails so `protect` can fail clearly when requested strength/quality floors are not met
- Added a collective subject-set batch mode so AuraLock can optimize one benchmark run across a full published split instead of only protecting files independently
- Added a benchmark runner to compare profiles over files/directories and aggregate results by profile
- Added a DreamBooth/LoRA benchmark harness with strict preflight, a clean baseline job, manifest generation, and execution planning
- Added a Docker GPU benchmark runtime with workspace path mapping, GPU smoke tests, and a dedicated container image for LoRA execution
- Added a Colab free-GPU notebook that installs AuraLock benchmark deps, clones Anti-DreamBooth, mounts Google Drive, and runs dry-run before execute

## Remaining Gaps Before "Full Product"

### High priority
- Add structured logging and per-run metadata export
- Verify Docker GPU benchmark runtime on a machine with Docker + NVIDIA access
- Add machine-readable report export (`csv`) and richer run history aggregation

### Medium priority
- Add benchmark datasets and repeatable evaluation reports
- Add release/version automation and changelog generation
- Add stronger CLI coverage for `protect` and `analyze` end-to-end flows

### Research priority
- Benchmark against actual LoRA / DreamBooth / style-mimicry pipelines
- Add purification robustness tests against external repos such as `robust-style-mimicry`

## Why This Matters

`MiroFish` looks professional because it combines code with deployment, environment setup, workflows, and clearer system boundaries. AuraLock is still smaller in scope, but it now has a cleaner runtime boundary, real batch/CLI behavior, deployment scaffolding, and a default protection mode aligned with the actual artist-protection goal instead of remaining just a classifier-attack demo.
