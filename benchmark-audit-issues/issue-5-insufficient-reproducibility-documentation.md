# Issue: Insufficient Reproducibility Documentation - Cannot Verify or Reproduce Benchmark Results

## Labels
`benchmark`, `reproducibility`, `documentation`, `research`, `critical`

## Problem Description

The repository provides **minimal documentation for reproducing benchmark results**. Critical information is missing: exact datasets used, model versions, random seeds, hardware specifications, environment details, and execution logs. Published results in README cannot be independently verified or reproduced, violating fundamental scientific principles.

## What is Wrong with the Benchmark

### 1. README Results Lack Reproducibility Information

**Published results** (README lines 52-58):
```markdown
| Run | Protection Score | PSNR | SSIM | Notes |
|-----|------------------|------|------|-------|
| `balanced` | `42.1` | `36.24` | `0.9346` | better visual quality, good study baseline |
| `subject` | `51.5` | `30.53` | `0.8270` | stronger drift for subject-style protection experiments |
| `fortress` | `53.2` | `29.08` | `0.7858` | more aggressive, visibly harsher output |
| `blindfold` | `61.1` | `26.53` | `0.6114` | strongest current anti-readability preset, largest fidelity cost |
| `collective n000050 / set_B` | `22.8` avg | `37.78` avg | `0.9666` avg | correct benchmark direction, objective still needs tuning |
```

**Missing critical information**:
- ❌ **Dataset**: Which image(s) were these scores measured on?
- ❌ **Image properties**: Resolution, format, content type?
- ❌ **Model versions**: Which ResNet18 weights? PyTorch version?
- ❌ **Random seed**: Was any randomness involved? What seed?
- ❌ **Hardware**: CPU or GPU? Which device?
- ❌ **Timestamp**: When were these results generated?
- ❌ **Reproducibility script**: How to regenerate these exact numbers?
- ❌ **Full report**: Where are the complete JSON reports?

**Cannot verify**:
- Did these results come from held-out test set or training data?
- Were results cherry-picked from multiple runs?
- Can independent researchers reproduce these exact values?

### 2. No Benchmark Result Archive

**Expected structure**:
```
Lock-ART./
+-- benchmark_results/
    +-- 2024-01-15_balanced_profile/
    |   +-- config.json          # Exact configuration
    |   +-- dataset_manifest.json # Images used
    |   +-- environment.json     # Versions, hardware
    |   +-- results.json         # Full metrics
    |   +-- logs/                # Execution logs
    |   +-- reproduce.sh         # Reproduction script
    +-- 2024-02-20_fortress_validation/
        +-- ...
```

**Actual state**:
```bash
$ ls benchmark_results/
# (directory does not exist)

$ find . -name "*results.json" -o -name "*benchmark*.json"
# (no archived results)
```

**No historical results are archived for verification.**

### 3. Incomplete Configuration Capture

**Benchmark reports** (`/src/auralock/services/protection.py:19-72`):
```python
@dataclass
class BenchmarkSummary:
    input_path: Path
    image_count: int
    entries: list[BenchmarkEntry]
    profile_summaries: dict[str, ProfileAggregate]

    # Missing:
    # - environment_info: Python version, PyTorch version, CUDA version
    # - model_info: ResNet18 weights version, model hash
    # - random_seed: Seed used for any stochastic operations
    # - execution_metadata: start time, duration, hardware used
    # - reproducibility_hash: Hash of all parameters for verification
```

**Current reports don't capture enough information to reproduce results.**

### 4. No Dataset Provenance Tracking

For the "collective n000050 / set_B" result:
```markdown
| `collective n000050 / set_B` | `22.8` avg | `37.78` avg | `0.9666` avg |
```

**Questions without answers**:
- Where is the `n000050` dataset?
- Is it publicly available?
- How many images in set_B? What content?
- What is the image resolution?
- Is this from Anti-DreamBooth dataset? Which version?

**Anti-DreamBooth reference** (`/src/auralock/benchmarks/antidreambooth.py:22-27`):
```python
DEFAULT_ANTI_DREAMBOOTH_TRAIN_SCRIPT = Path(
    ".cache_ref/Anti-DreamBooth/train_dreambooth.py"
)
```

**But**:
- `.cache_ref/` is not in repository
- No instructions on how to obtain this dataset
- No documentation on dataset structure or licensing
- No manifest listing actual images used

### 5. Model Weights and Versions Not Documented

**ResNet18 loading** (`/src/auralock/core/style.py:107`):
```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)
```

**Questions**:
- What is `ResNet18_Weights.DEFAULT` at time of benchmark?
- Which torchvision version? (weights change across versions)
- What is the model hash/checksum?
- Could torchvision updates change benchmark results?

**Lack of version pinning creates reproducibility risk.**

### 6. Random Seed Usage Inconsistent

**Some operations use seeds**:
```python
# /src/auralock/benchmarks/lora.py:74
seed: int = 42  # Default seed for LoRA training
```

**But unclear if seed controls all randomness**:
- StyleCloak optimization: Is it deterministic? Uses which RNG?
- Feature extraction: Any dropout or stochastic layers?
- Data loading: Iteration order deterministic?

**No global `set_seed()` function to ensure full reproducibility.**

### 7. Hardware and Performance Variability Not Addressed

**No documentation on**:
- CPU vs GPU differences in numerical precision
- Expected runtime per profile (for verification)
- Memory requirements
- Whether results vary across hardware (floating point differences)

**Example concern**: PyTorch operations can produce slightly different results on CPU vs GPU due to floating point precision. Is this variance documented?

## Why This Can Mislead Users

### Scenario 1: Cannot Verify Published Claims

Independent researcher wants to verify "balanced" profile achieves PSNR=36.24:

```bash
# Attempt 1: No dataset information
$ auralock protect ??? -o protected.png --profile balanced
# What input image? Where to get it?

# Attempt 2: Try random image
$ auralock protect my_artwork.png -o protected.png --profile balanced --report report.json
$ cat report.json
# PSNR: 38.15  ← Different from published 36.24

# Questions:
# - Is my result wrong?
# - Was published result cherry-picked?
# - Are we using different image/configuration?
# - No way to know.
```

**Cannot verify published results → Cannot trust the benchmark.**

### Scenario 2: False Confidence from Unverifiable Results

User sees impressive metrics in README:
```
| `fortress` | `53.2` | `29.08` | `0.7858` |
```

User assumes:
- "These results were rigorously validated"
- "Independent researchers verified these numbers"
- "The methodology is reproducible"

Reality:
- No independent verification possible (missing reproduction info)
- Could be cherry-picked from favorable runs
- Could be from overfit training data
- **No way to know → Blind trust required**

### Scenario 3: Wasted Effort on Non-Reproducible Research

Researcher builds on AuraLock results:
```python
# Paper: "Building on AuraLock's fortress profile (protection_score=53.2)..."
# Attempts to reproduce baseline
# Cannot match published results (gets 48.1 instead of 53.2)
# Wastes weeks debugging
# Eventually gives up or publishes with inconsistent baseline
```

**Lack of reproducibility wastes research effort and undermines trust in field.**

### Scenario 4: Version Drift Breaks Comparisons

Timeline:
```
2024-01: Published results using PyTorch 2.0, torchvision 0.15
2024-06: User installs AuraLock with PyTorch 2.2, torchvision 0.17
2024-06: User cannot reproduce published results
2024-06: ResNet18 weights changed in torchvision 0.17
2024-06: All comparisons invalidated, no documentation to understand why
```

**Without version pinning and documentation, results become unverifiable over time.**

## Evidence from Repository

### 1. No Archived Benchmark Results

```bash
$ find /home/runner/work/Lock-ART./Lock-ART. -name "*benchmark*results*" -o -name "results.json"
# (no matches)

$ ls benchmark_results/
# (does not exist)
```

**No historical benchmark results are stored in repository.**

### 2. Incomplete Report Metadata

Examine report structure (`/src/auralock/services/protection.py:55-72`):
```python
def to_report_dict(self) -> dict[str, object]:
    return {
        "input_path": str(self.input_path),
        "image_count": self.image_count,
        # ... metrics ...

        # Missing:
        # "environment": {"python": "3.11", "pytorch": "2.1.0", "cuda": "11.8"},
        # "timestamp": "2024-01-15T10:30:00Z",
        # "hardware": {"device": "cpu", "cpu_model": "Intel i7"},
        # "random_seed": 42,
        # "reproducibility_hash": "a3f5b9c...",
    }
```

**Reports don't include environment metadata needed for reproduction.**

### 3. Version Pinning is Weak

`pyproject.toml` dependencies (lines 23-29):
```toml
[project]
dependencies = [
    "torch>=2.0.0",        # ⚠️ >=2.0.0 allows 2.0, 2.1, 2.2, 2.3...
    "torchvision>=0.15.0", # ⚠️ Weights may differ across versions
    "scikit-image>=0.20.0",
    # ...
]
```

**Loose version constraints allow dependency drift**, breaking reproducibility over time.

### 4. No Reproducibility Guide

Documentation search:
```bash
$ grep -r "reproduc" docs/
# (minimal matches, no reproduction guide)

$ ls docs/REPRODUCIBILITY.md
# (does not exist)
```

**No dedicated guide for reproducing benchmark results.**

### 5. README Results Have No Source Trace

README lines 52-58 show results but:
```bash
$ grep -r "42.1\|36.24\|0.9346" .
# README.md:54:| `balanced` | `42.1` | `36.24` | `0.9346` |
# (only in README, no source data)
```

**Published results exist only in README markdown**, not backed by archived data files.

### 6. Colab Notebook Has No Executed Outputs

`/notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb`:
- Designed for benchmark execution on GPU
- **But notebook has no executed cell outputs**
- Cannot see example results or validate notebook works
- Users must blindly trust notebook will produce claimed results

## Proposed Benchmark Upgrade

### Phase 1: Reproducibility Metadata in Reports

Extend all reports to include full reproducibility information:
```python
# /src/auralock/core/reproducibility.py

import platform
import torch
import sys
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EnvironmentInfo:
    """Complete environment information for reproducibility."""
    timestamp: str
    python_version: str
    pytorch_version: str
    pytorch_cuda_version: str | None
    torchvision_version: str
    numpy_version: str
    platform: str
    cpu_model: str
    gpu_model: str | None
    device_used: str
    random_seed: int | None

    @classmethod
    def capture(cls, device: str, random_seed: int | None = None) -> 'EnvironmentInfo':
        """Capture current environment information."""
        import numpy as np
        import torchvision
        import cpuinfo

        return cls(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            python_version=sys.version,
            pytorch_version=torch.__version__,
            pytorch_cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            torchvision_version=torchvision.__version__,
            numpy_version=np.__version__,
            platform=platform.platform(),
            cpu_model=cpuinfo.get_cpu_info()['brand_raw'],
            gpu_model=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            device_used=device,
            random_seed=random_seed,
        )


@dataclass
class ModelInfo:
    """Model version and checksum for reproducibility."""
    model_name: str
    weights_name: str
    weights_url: str
    weights_hash: str  # SHA256 of model weights

    @classmethod
    def from_feature_extractor(cls, extractor) -> 'ModelInfo':
        """Extract model information from feature extractor."""
        # Implementation to get model metadata
        ...


@dataclass
class DatasetInfo:
    """Dataset provenance for reproducibility."""
    dataset_name: str
    dataset_version: str | None
    image_paths: list[str]
    image_hashes: dict[str, str]  # path -> SHA256 hash
    split_type: str | None  # "train", "val", "test"
    dataset_manifest_url: str | None

    @classmethod
    def from_paths(cls, paths: list[Path], dataset_name: str) -> 'DatasetInfo':
        """Create dataset info from image paths."""
        import hashlib

        image_hashes = {}
        for path in paths:
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            image_hashes[str(path)] = file_hash

        return cls(
            dataset_name=dataset_name,
            dataset_version=None,
            image_paths=[str(p) for p in paths],
            image_hashes=image_hashes,
            split_type=None,
            dataset_manifest_url=None,
        )


@dataclass
class ReproducibilityBundle:
    """Complete bundle for reproducing benchmark results."""
    environment: EnvironmentInfo
    model: ModelInfo
    dataset: DatasetInfo
    config: dict  # All configuration parameters
    reproducibility_hash: str  # Hash of all above for quick verification

    def save(self, output_path: Path):
        """Save reproducibility bundle to JSON."""
        with open(output_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> 'ReproducibilityBundle':
        """Load reproducibility bundle from JSON."""
        with open(input_path) as f:
            data = json.load(f)
        return cls(**data)
```

Update all reports to include reproducibility bundle:
```python
@dataclass
class BenchmarkSummary:
    # ... existing fields ...
    reproducibility: ReproducibilityBundle  # NEW
```

### Phase 2: Archived Benchmark Results

Create systematic result archiving:
```
Lock-ART./
+-- benchmark_results/
    +-- README.md                              # Index of all results
    +-- 2024-03-31_balanced_validation/
    |   +-- config.json                        # Profile configuration
    |   +-- reproducibility.json               # Full environment info
    |   +-- dataset_manifest.json              # Images used with hashes
    |   +-- results.json                       # Full metrics
    |   +-- reproduce.sh                       # Exact reproduction command
    |   +-- protected_samples/                 # Sample protected images
    |   |   +-- artwork_001_protected.png
    |   |   +-- artwork_002_protected.png
    |   +-- logs/
    |       +-- execution.log                  # Full execution log
    +-- 2024-03-31_fortress_validation/
        +-- ...
```

Add command to archive results:
```bash
$ auralock benchmark artwork.png \
    --profile balanced \
    --report report.json \
    --archive-results benchmark_results/2024-03-31_balanced_validation

✓ Results archived to benchmark_results/2024-03-31_balanced_validation/
✓ Reproducibility bundle saved
✓ Reproduction script generated: reproduce.sh
```

### Phase 3: Reproducibility Verification Tool

Create verification tool:
```bash
# Verify archived results match published claims
$ auralock verify-results benchmark_results/2024-03-31_balanced_validation/

Checking reproducibility...
✓ Environment matches (Python 3.11.0, PyTorch 2.1.0)
✓ Model weights match (SHA256: a3f5b9c...)
✓ Dataset manifest valid (10 images, hashes verified)
✓ Configuration matches archived config
✓ Re-running benchmark...
✓ Results match within tolerance (PSNR: 36.24 ± 0.05)

Verification: PASSED
```

### Phase 4: Pinned Reproducibility Environment

Create locked dependency file:
```bash
# Generate exact version lock
$ pip freeze > requirements-lock.txt

# Or use Poetry
$ poetry lock
```

Add reproducibility environment setup:
```bash
# scripts/setup_reproducibility_env.sh
#!/bin/bash
# Create exact environment for reproducing benchmark results

python -m venv venv_reproduce
source venv_reproduce/bin/activate
pip install -r requirements-lock-2024-03-31.txt

echo "Reproducibility environment ready"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
```

### Phase 5: Benchmark Result Registry

Create `benchmark_results/README.md`:
```markdown
# AuraLock Benchmark Results Registry

All official benchmark results with full reproducibility information.

## Published Results

| Date       | Profile   | PSNR  | SSIM   | Protection Score | Archive Link | Status |
|------------|-----------|-------|--------|------------------|--------------|--------|
| 2024-03-31 | balanced  | 36.24 | 0.9346 | 42.1             | [2024-03-31_balanced_validation/](./2024-03-31_balanced_validation/) | ✓ Verified |
| 2024-03-31 | fortress  | 29.08 | 0.7858 | 53.2             | [2024-03-31_fortress_validation/](./2024-03-31_fortress_validation/) | ✓ Verified |

## Verification Status

- ✓ Verified: Results independently reproduced with matching metrics
- ⚠️ Pending: Not yet independently verified
- ❌ Failed: Reproduction attempt failed or results don't match

## How to Reproduce

```bash
# Clone repository
git clone https://github.com/VoDaiLocz/Lock-ART.

# Navigate to result directory
cd benchmark_results/2024-03-31_balanced_validation/

# Setup reproducibility environment
bash setup_env.sh

# Run reproduction script
bash reproduce.sh

# Verify results match
auralock verify-results .
```
```

### Phase 6: Documentation Updates

Create `docs/REPRODUCIBILITY_GUIDE.md`:
```markdown
# Reproducibility Guide

## Reproducing Published Results

All benchmark results in README are backed by archived data in `benchmark_results/`.

### Step 1: Find Result Archive
```bash
# Check registry
cat benchmark_results/README.md

# Navigate to specific result
cd benchmark_results/2024-03-31_balanced_validation/
```

### Step 2: Inspect Reproducibility Info
```bash
# Check environment requirements
cat reproducibility.json

# Check dataset manifest
cat dataset_manifest.json

# Check exact configuration
cat config.json
```

### Step 3: Setup Environment
```bash
# Create exact environment used for original results
bash setup_env.sh
source venv_reproduce/bin/activate
```

### Step 4: Run Reproduction
```bash
# Execute reproduction script
bash reproduce.sh

# Compare results
diff results.json ../original_results.json
```

## Creating Reproducible Benchmarks

When publishing new benchmark results:

1. Use `--archive-results` flag
2. Commit archived results to repository
3. Verify results are reproducible
4. Update benchmark registry
5. Update README with links to archived results
```

## Acceptance Criteria

### Phase 1: Metadata Capture (Week 1)
- [ ] Implement `EnvironmentInfo`, `ModelInfo`, `DatasetInfo` classes
- [ ] Update all reports to include reproducibility bundle
- [ ] Test metadata capture on various platforms

### Phase 2: Result Archiving (Week 1-2)
- [ ] Create `benchmark_results/` directory structure
- [ ] Implement `--archive-results` CLI flag
- [ ] Archive all currently published results with full metadata
- [ ] Generate reproduction scripts for each archived result

### Phase 3: Verification Tool (Week 2)
- [ ] Implement `auralock verify-results` command
- [ ] Test verification on archived results
- [ ] Document tolerance levels for numerical differences

### Phase 4: Pinned Dependencies (Week 2)
- [ ] Generate `requirements-lock.txt` for current version
- [ ] Create reproducibility environment setup script
- [ ] Document version pinning in contributing guide

### Phase 5: Registry and Documentation (Week 3)
- [ ] Create `benchmark_results/README.md` registry
- [ ] Create `docs/REPRODUCIBILITY_GUIDE.md`
- [ ] Update main README to link to archived results
- [ ] Add reproducibility checklist to PR template

### Phase 6: Independent Verification (Ongoing)
- [ ] Invite independent researchers to verify results
- [ ] Mark verification status in registry
- [ ] Address any reproducibility issues discovered
- [ ] Maintain registry with all verification attempts

## Additional Context

### Scientific Reproducibility Standards

From Nature's reproducibility checklist:
1. **Code availability**: ✓ (repository is public)
2. **Data availability**: ❌ (datasets not archived)
3. **Environment specification**: ❌ (loose version constraints)
4. **Random seeds**: ⚠️ (partial documentation)
5. **Detailed methodology**: ⚠️ (incomplete)

**AuraLock needs significant improvements to meet basic reproducibility standards.**

### ACM Artifact Evaluation Badges

ACM awards reproducibility badges:
- **Artifacts Available**: Code and data publicly accessible
- **Artifacts Evaluated - Functional**: Documented, complete, exercisable
- **Results Reproduced**: Independent verification succeeded

**Current state**: Would not qualify for any badge without these improvements.

### Cost of Poor Reproducibility

Research community impact:
- Wasted effort attempting to reproduce results
- Inability to build on prior work
- Erosion of trust in benchmark claims
- Difficulty comparing methods across papers

**Strong reproducibility infrastructure is essential for credible research.**

## References

- README.md lines 52-58 - Unverifiable benchmark results
- `/src/auralock/services/protection.py:19-72` - Incomplete report metadata
- `/notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb` - No executed outputs
- Nature [Reproducibility Checklist](https://www.nature.com/documents/nr-reporting-summary-flat.pdf)
- ACM [Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- [Papers with Code reproducibility best practices](https://paperswithcode.com/about)
