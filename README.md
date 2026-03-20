# AuraLock

<p align="center">
  <strong>A learning-focused artwork cloaking toolkit for anti-mimicry experiments.</strong><br/>
  Study single-image protection, directory workflows, and Anti-DreamBooth-aligned subject-set benchmarking from one repository.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-75%20passed-16a34a" alt="75 tests passed" />
  <img src="https://img.shields.io/badge/version-0.1.0-2563eb" alt="Version 0.1.0" />
  <img src="https://img.shields.io/badge/package-auralock-0f766e" alt="Package auralock" />
  <img src="https://img.shields.io/badge/release-not%20tagged-lightgrey" alt="Release not tagged" />
  <img src="https://img.shields.io/badge/status-alpha-f59e0b" alt="Alpha status" />
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-2563eb" alt="Python 3.10 to 3.12" />
  <img src="https://img.shields.io/badge/license-MIT-eab308" alt="MIT License" />
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-usage">CLI</a> •
  <a href="#profiles">Profiles</a> •
  <a href="#benchmark-snapshot">Benchmarks</a> •
  <a href="#tested-environment">Tested Environment</a>
</p>

## Overview

**AuraLock** is a study repository for artwork cloaking and anti-mimicry evaluation. It does not claim that all vision models can be permanently blocked. The goal is narrower and more honest:

- keep images usable for humans
- increase the cost of feature extraction or style mimicry workflows
- benchmark those trade-offs with repeatable reports instead of subjective demos

The hero visual below is rendered from a local HTML asset in this repository through Playwright.

<p align="center">
  <img src="docs/assets/readme-overview.png" alt="AuraLock hero overview" width="100%" />
</p>

## Project Snapshot

| Area | Current state |
|------|---------------|
| **Core engine** | `StyleCloak` with `safe`, `balanced`, `strong`, `subject`, `fortress`, and `blindfold` profiles |
| **Operational modes** | single image, directory batch, adaptive guardrails, collective subject-set batch |
| **Benchmarking** | profile benchmark, LoRA manifest planning, Anti-DreamBooth split harness, Docker GPU path, Colab notebook |
| **Interfaces** | CLI, optional Gradio UI, JSON reports |
| **Project status** | learning project / alpha package metadata, not a finished commercial release |

## Quick Start

```bash
git clone https://github.com/VoDaiLocz/Lock-ART.
cd Lock-ART.

python -m venv .venv
.\.venv\Scripts\activate

pip install -e ".[dev]"
```

Optional extras:

```bash
# Web UI
pip install -e ".[ui,dev]"

# DreamBooth / LoRA benchmark dependencies
pip install -e ".[benchmark,dev]"
```

## CLI Usage

```bash
# Protect a single image
auralock protect artwork.png -o protected.png

# Protect with a stronger profile and save a JSON report
auralock protect artwork.png -o protected.png --profile strong --report reports/protect.json

# Adaptive mode: escalate profiles until thresholds are met or fail clearly
auralock protect artwork.png -o protected.png ^
  --auto-profiles safe,balanced,strong,subject,fortress,blindfold ^
  --min-protection-score 25 ^
  --min-ssim 0.92 ^
  --min-psnr 35 ^
  --report reports/protect-adaptive.json

# Aggressive anti-readability mode
auralock protect artwork.png -o protected-blindfold.png --profile blindfold

# Protect an entire directory
auralock batch ./artworks ./protected --recursive

# Collective subject-set protection
auralock batch ./.cache_ref/Anti-DreamBooth/data/n000050/set_B ./protected_subject ^
  --profile subject ^
  --collective ^
  --working-size 384 ^
  --report reports/batch-collective.json

# Compare original and protected images
auralock analyze original.png protected.png --report reports/analyze.json

# Compare multiple profiles on one file or directory
auralock benchmark artwork.png --profiles safe,balanced,strong --report reports/benchmark.json

# Create a DreamBooth / LoRA benchmark manifest
auralock benchmark-lora ./artworks ^
  --work-dir benchmark_runs/lora ^
  --pretrained-model-path ./stable-diffusion/stable-diffusion-1-5 ^
  --script-path ./.cache_ref/Anti-DreamBooth/train_dreambooth_lora.py ^
  --instance-prompt "a sks painting" ^
  --class-prompt "a painting" ^
  --report reports/lora-manifest.json

# Plan a benchmark aligned with Anti-DreamBooth subject splits
auralock benchmark-antidreambooth ./.cache_ref/Anti-DreamBooth/data/n000050 ^
  --profiles safe ^
  --pretrained-model-path ./stable-diffusion/stable-diffusion-2-1-base ^
  --report reports/antidreambooth-manifest.json

# Optional web UI
auralock webui --host 127.0.0.1 --port 7860
```

## Python API

```python
from auralock import ProtectionService
from auralock.core.image import save_image

service = ProtectionService()
result = service.protect_file("artwork.png", profile="balanced")

save_image(result.protected_tensor, "protected.png")
print(result.quality_report)
print(result.protection_report)
print(result.to_report_dict(output_path="protected.png"))
```

## Profiles

| Profile | Goal | Default direction |
|---------|------|-------------------|
| `safe` | Prioritize visual quality | low epsilon, light drift |
| `balanced` | Balance quality and protection | best general-purpose learning baseline |
| `strong` | Push protection harder | higher drift, lower fidelity |
| `subject` | Stronger preset for subject-style experiments | closer to Anti-DreamBooth-style runs |
| `fortress` | Maximize protection within the current proxy approach | visible image changes |
| `blindfold` | Aggressive anti-readability mode | strongest local score, harshest visual trade-off |

Adaptive CLI mode saves the output and report even when thresholds are missed, but returns a non-zero exit code so an automation pipeline does not mistake a weak result for a success.

## Benchmark Snapshot

Current local report highlights:

| Run | Protection Score | PSNR | SSIM | Notes |
|-----|------------------|------|------|-------|
| `balanced` | `42.1` | `36.24` | `0.9346` | better visual quality, good study baseline |
| `subject` | `51.5` | `30.53` | `0.8270` | stronger drift for subject-style protection experiments |
| `fortress` | `53.2` | `29.08` | `0.7858` | more aggressive, visibly harsher output |
| `blindfold` | `61.1` | `26.53` | `0.6114` | strongest current anti-readability preset, largest fidelity cost |
| `collective n000050 / set_B` | `22.8` avg | `37.78` avg | `0.9666` avg | correct benchmark direction, objective still needs tuning |

The `Protection Score` is an internal proxy derived from embedding and style similarity after robustness transforms. It is useful for comparisons inside this repository, not as a universal guarantee against every AI system.

## Workflow

```mermaid
flowchart LR
    A["Original artwork"] --> B["AuraLock CLI / service layer"]
    B --> C["Single image protect"]
    B --> D["Batch protect"]
    B --> E["Collective subject protect"]
    C --> F["Protected output + JSON report"]
    D --> F
    E --> G["Published split for benchmark"]
    G --> H["DreamBooth / LoRA benchmark harness"]
    H --> I["Metrics, manifests, and review"]
```

## Tested Environment

This repository has been exercised locally on the following machine:

| Item | Value |
|------|-------|
| OS | `Windows 10 Pro 64-bit (10.0.19045)` |
| CPU | `Intel Core i7-6600U @ 2.60GHz` |
| CPU topology | `2 cores / 4 threads` |
| RAM | `8.4 GB` installed |
| GPU | `Intel HD Graphics 520` |
| Local Python | `3.12.10` |

What runs on this machine:

- single-image protection on CPU
- directory batch protection on CPU
- analysis and report generation
- README asset generation with Playwright

What does **not** run well on this machine:

- real DreamBooth / LoRA fine-tuning
- CUDA benchmark execution
- Docker GPU benchmark runtime

For actual DreamBooth or LoRA execution, use a CUDA-capable NVIDIA GPU or the provided Colab / Docker benchmark path.

## Project Structure

```text
Lock-ART./
├── .github/workflows/
├── docs/
├── notebooks/
├── src/
│   ├── auralock/
│   │   ├── attacks/
│   │   ├── benchmarks/
│   │   ├── core/
│   │   ├── services/
│   │   ├── ui/
│   │   └── cli.py
│   └── tests/
├── Dockerfile
├── Dockerfile.benchmark
├── docker-compose.yml
├── docker-compose.benchmark.yml
└── pyproject.toml
```

## Documentation

- [Product Audit](docs/PRODUCT_AUDIT.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Research Roadmap](docs/RESEARCH_ROADMAP.md)
- [Colab Benchmark Notebook](notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb)

## Verification

```bash
pytest -q
ruff check src
black --check src
```

## Release and Package Notes

- The package metadata currently lives in [`pyproject.toml`](pyproject.toml) as `auralock` version `0.1.0`.
- No public GitHub release is tagged yet.
- The repository should be treated as an alpha learning project, not a finished commercial product.

## License

This project is distributed under the **MIT License**. See [LICENSE](LICENSE).

## Author

**locfaker**

- GitHub: [@locfaker](https://github.com/locfaker)
