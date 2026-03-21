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

## Audit: Test Coverage Gap Issues

### 1) Regression baseline coverage for protection outputs
- **Missing test coverage area:** No stable regression baselines for `protect` output metrics (quality/style/readability) across code changes.
- **Risk if not covered:** Silent behavior drift can change protection strength or image quality without detection.
- **Suggested test strategy:** Add deterministic fixture inputs and snapshot-style assertions for key report values with bounded tolerances per profile.
- **Acceptance criteria:** CI fails when baseline metrics move outside agreed tolerance bands for the same fixture/profile combination.
- **Labels:** `testing`, `regression`, `quality`

### 2) Cross-module integration flow validation
- **Missing test coverage area:** Incomplete end-to-end flow tests covering `load -> protect -> analyze -> report/save` across service and CLI boundaries.
- **Risk if not covered:** Individually passing unit tests can still hide broken integration contracts and malformed outputs.
- **Suggested test strategy:** Add integration tests that execute full workflows on sample images and assert final artifacts + report structure.
- **Acceptance criteria:** A single integration suite validates expected files, report keys, and non-empty analysis metrics for successful runs.
- **Labels:** `testing`, `integration`, `cli`

### 3) CLI end-to-end command matrix expansion
- **Missing test coverage area:** Limited end-to-end coverage for `protect`, `analyze`, `demo`, `webui`, and argument/profile combinations.
- **Risk if not covered:** User-facing commands can regress even if internal APIs remain correct.
- **Suggested test strategy:** Use `CliRunner` to cover happy-path and invalid-argument matrices, including profile presets and report options.
- **Acceptance criteria:** Each public CLI command has at least one passing end-to-end test and one explicit failure-path test.
- **Labels:** `testing`, `cli`, `e2e`

### 4) Docker workflow runtime validation
- **Missing test coverage area:** Docker assets exist, but tests do not validate container build/run workflows for application and benchmark images.
- **Risk if not covered:** Published images may build but fail at runtime due to dependency, entrypoint, or path-mapping issues.
- **Suggested test strategy:** Add CI job(s) to build both Dockerfiles and run smoke commands (`auralock --help`, benchmark preflight) in containers.
- **Acceptance criteria:** CI verifies image build + smoke execution for `Dockerfile` and `Dockerfile.benchmark`.
- **Labels:** `testing`, `docker`, `ci`

### 5) Release/package verification checks
- **Missing test coverage area:** No automated validation for wheel/sdist creation, installability, script entry points, and version metadata consistency.
- **Risk if not covered:** Broken releases can be published with unusable artifacts or mismatched package metadata.
- **Suggested test strategy:** Add release-gate workflow to build artifacts, install from built wheel in a clean env, and verify `auralock --version`.
- **Acceptance criteria:** Release CI must pass package build, install, and CLI smoke checks before publish steps proceed.
- **Labels:** `testing`, `release`, `packaging`

### 6) Benchmark reproducibility assertions
- **Missing test coverage area:** Benchmark tests emphasize planning/manifests but do not assert reproducibility under fixed seeds/configuration.
- **Risk if not covered:** Benchmark comparisons may be noisy or non-repeatable, reducing confidence in reported improvements.
- **Suggested test strategy:** Add reproducibility tests that run benchmark routines twice with fixed settings and compare summary metrics/manifests.
- **Acceptance criteria:** Repeated benchmark runs under fixed seed/config produce matching manifest content and stable summary outputs.
- **Labels:** `testing`, `benchmark`, `reproducibility`

### 7) Failure-path and resiliency coverage
- **Missing test coverage area:** Partial failure-path tests; gaps remain for file I/O permissions, missing/corrupt inputs, and partial batch failures.
- **Risk if not covered:** Real-world errors can produce unclear messages, silent skips, or incomplete outputs.
- **Suggested test strategy:** Add explicit negative tests for invalid files, write failures, and per-file batch error handling with assertive messaging.
- **Acceptance criteria:** Failure-path tests assert non-zero exit behavior (or controlled continuation) and clear user-facing error diagnostics.
- **Labels:** `testing`, `reliability`, `error-handling`

### 8) Image output validation after save/load roundtrips
- **Missing test coverage area:** Current image tests are lightweight; gaps remain in post-save/load validation for shape, dtype, bounds, and format fidelity.
- **Risk if not covered:** Saved outputs may degrade or become invalid while tests still pass.
- **Suggested test strategy:** Add roundtrip tests per supported extension validating size, channel layout, dtype/range, and expected tolerance envelopes.
- **Acceptance criteria:** Save/load tests cover supported formats and verify output invariants + bounded pixel drift where lossy encoding applies.
- **Labels:** `testing`, `image-io`, `validation`

## Why This Matters

`MiroFish` looks professional because it combines code with deployment, environment setup, workflows, and clearer system boundaries. AuraLock is still smaller in scope, but it now has a cleaner runtime boundary, real batch/CLI behavior, deployment scaffolding, and a default protection mode aligned with the actual artist-protection goal instead of remaining just a classifier-attack demo.
