# Issue: Missing Ground-Truth LoRA/DreamBooth Validation - Benchmark Infrastructure Exists But Unused

## Labels
`benchmark`, `critical`, `validation`, `research`, `missing-feature`

## Problem Description

The repository contains complete infrastructure for real-world LoRA/DreamBooth mimicry benchmarking (`/src/auralock/benchmarks/lora.py`, `/src/auralock/benchmarks/antidreambooth.py`) **but provides zero published validation results**. All protection claims are based on proxy metrics without demonstrating that the protection actually prevents style mimicry when evaluated against real generative models.

## What is Wrong with the Benchmark

### 1. Infrastructure vs Results Gap

**Infrastructure Present**:
```python
# /src/auralock/benchmarks/lora.py:239-442
class LoraBenchmarkHarness:
    """Prepare and optionally execute a real DreamBooth/LoRA benchmark workflow."""

    def run(self, ..., execute: bool = False, ...):
        # Can prepare protected datasets
        # Can build training commands
        # Can execute real LoRA training
        # Can run inference tests
```

**Reality**:
- `execute=False` by default (line 320)
- No published results in README
- No validation data in repository
- All claims based on proxy metrics

### 2. Current "Benchmark" is Actually Metric Comparison

What the repository calls "benchmark" (`auralock benchmark`):
```python
# /src/auralock/services/protection.py:172-234
def benchmark_file(...):
    # Compare PSNR, SSIM, protection_score across profiles
    # Does NOT measure actual mimicry prevention
    # Returns proxy metrics only
```

This is **profile comparison**, not **mimicry prevention validation**.

### 3. No Evidence That Protection Works

README claims (lines 52-58):
```
| `fortress` | `53.2` | `29.08` | `0.7858` | more aggressive, visibly harsher output |
```

**Questions without answers**:
- Can DreamBooth still learn style from "fortress" protected images? **Unknown**
- How many LoRA training runs succeed vs fail after protection? **Not measured**
- What is the success rate degradation: clean (baseline) vs protected? **No data**

### 4. Preflight System Exists But No Follow-Through

```python
# /src/auralock/benchmarks/lora.py:107-176
def evaluate_lora_preflight(...):
    """Check whether the current machine can run a real LoRA benchmark."""
    # Checks CUDA availability
    # Validates required modules (diffusers, accelerate, transformers, peft)
    # Validates script paths and model directories
    # Returns ready/not_ready status
```

**But**: The repository never publishes results from machines where `preflight.ready == True`.

## Why This Can Mislead Users

### Scenario 1: Unvalidated Protection Claims

User reads README:
```
| `blindfold` | `61.1` | `26.53` | `0.6114` | strongest current anti-readability preset |
```

User assumes:
- "61.1 protection score means DreamBooth can't learn my art style"
- "This has been validated against real mimicry models"

Reality:
- No validation against actual LoRA/DreamBooth training
- The "61.1" is just feature drift in ResNet18 space
- Real-world effectiveness is **completely unknown**

### Scenario 2: False Security

Artists deploy protection in production based on:
- High proxy scores
- Professional-looking benchmark infrastructure in code
- Assumption that "if they built it, they must have tested it"

Meanwhile:
- Zero published validation results
- No evidence protection survives actual mimicry training
- Infrastructure exists but is never executed with `execute=True`

### Scenario 3: Wasted Development Effort

Developers tune profiles to maximize protection_score without knowing:
- Does higher protection_score → lower mimicry success rate? **Unmeasured correlation**
- Which profile actually prevents mimicry best? **No comparative validation**
- Are visual quality trade-offs worthwhile? **No evidence of actual protection benefit**

## Evidence from Repository

### 1. Benchmark Infrastructure Exists

**LoRA Benchmark** (`/src/auralock/benchmarks/lora.py`):
- Lines 239-442: Full harness implementation
- Lines 179-219: Training command builder for `accelerate launch`
- Lines 222-236: Inference command builder
- Lines 44-80: Configuration dataclass with all hyperparameters

**Anti-DreamBooth Benchmark** (`/src/auralock/benchmarks/antidreambooth.py`):
- Lines 152-330: Subject split benchmark harness
- Lines 110-149: Paper-style set_A/set_B/set_C layout resolver
- Lines 22-27: Default scripts and prompts configured

**Docker Runtime** (`/src/auralock/benchmarks/docker_runtime.py`):
- GPU-accelerated containerized benchmark execution
- Supports distributed training with configurable GPU counts

### 2. But Zero Published Results

**Search for validation data**:
```bash
# No results directory
$ ls /home/runner/work/Lock-ART./Lock-ART./results
# (does not exist)

# No benchmark results in docs
$ grep -r "mimicry_success_rate" docs/
# (no matches)

# No published training logs
$ find . -name "*lora*results*" -o -name "*dreambooth*results*"
# (no matches)
```

**Test suite validates infrastructure but not outcomes**:
```python
# /src/tests/test_lora_benchmark.py
def test_lora_preflight_ready_state():
    # Tests that preflight detection works
    # Does NOT test that protection prevents mimicry

def test_lora_manifest_generation():
    # Tests manifest structure
    # Does NOT measure mimicry prevention
```

### 3. Colab Notebook Exists But No Results

`/notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb`:
- Designed for GPU execution on Google Colab
- **But no published outputs or results from the notebook**
- README mentions it (line 231) but doesn't link to results

### 4. Acknowledged Gap

README line 60:
> "It is useful for relative comparisons inside this repository, not as a universal guarantee against all AI systems."

RESEARCH_ROADMAP.md lines 48-51:
```
### Giai đoạn 3 (dài hạn: 2-3 tháng)
- Chạy benchmark thực tế trên GPU (LoRA/DreamBooth)
- Tổng hợp kết quả có kiểm định cơ bản
- Đề xuất hướng cải tiến objective dựa trên dữ liệu thực nghiệm
```

Translation: "Phase 3 (long-term: 2-3 months) - Run real GPU benchmarks, aggregate results, propose improvements based on experimental data."

**This is still in the roadmap, meaning it hasn't been done yet.**

## Proposed Benchmark Upgrade

### Phase 1: Minimal Validation Study

**Objective**: Generate first ground-truth results to validate proxy metrics.

**Protocol**:
```yaml
dataset:
  subjects: 10 diverse artworks (portraits, landscapes, abstract, anime)
  source: Public domain or author-owned images

profiles_to_test:
  - clean (unprotected baseline)
  - balanced
  - strong
  - fortress

mimicry_model:
  base_model: stabilityai/stable-diffusion-2-1-base
  method: DreamBooth LoRA
  training_steps: 400
  resolution: 512
  batch_size: 1
  learning_rate: 1e-4

evaluation_per_profile:
  1. Train LoRA model on protected images
  2. Generate 20 samples with trained model
  3. Measure mimicry success:
     - Human evaluation (3 raters): "Does output match original style?"
     - CLIP similarity: generated_samples vs original_style
     - FID score: generated_distribution vs original_distribution
  4. Report success rate: percentage of successful mimicry attempts

expected_output:
  - Table: profile → mimicry_success_rate
  - Analysis: correlation(protection_score, mimicry_prevention)
  - Recommendation: threshold recalibration
```

**Resource Requirements**:
- 10 subjects × 5 profiles = 50 training runs
- ~15-30 minutes per training run on V100 = 12-25 GPU hours
- Cost estimate: $25-50 on Google Colab Pro (affordable validation)

### Phase 2: Comprehensive Validation Suite

Expand validation to:
- **N=50 subjects** across diverse styles
- **Multiple mimicry methods**: DreamBooth, LoRA, Textual Inversion, IP-Adapter
- **Robustness testing**: Evaluate with common preprocessing (JPEG, resize, crop)
- **Comparison baseline**: Academic anti-mimicry methods (Anti-DreamBooth, Mist, Glaze if reproducible)

### Phase 3: Continuous Validation

Integrate ground-truth validation into CI/CD:
```yaml
# .github/workflows/validation.yml
- name: Monthly GPU Validation
  # Run subset of validation tests on GPU runner
  # Update results dashboard
  # Alert if protection effectiveness degrades
```

### Phase 4: Public Results Dashboard

Create `docs/VALIDATION_RESULTS.md`:
```markdown
# Ground-Truth Validation Results

## Last Updated: 2024-XX-XX

### Protection Effectiveness vs DreamBooth LoRA

| Profile   | Protection Score | Mimicry Success Rate | Samples |
|-----------|------------------|---------------------|---------|
| clean     | 0.0              | 95% (baseline)      | 50      |
| balanced  | 42.1             | 78% ⚠️              | 50      |
| strong    | 48.7             | 61%                 | 50      |
| fortress  | 53.2             | 45%                 | 50      |
| blindfold | 61.1             | 32%                 | 50      |

**Key Findings**:
- Correlation: R²=0.83 (strong correlation)
- But even "blindfold" allows 32% mimicry success
- Visual quality trade-off may not be worthwhile for moderate profiles
```

## Acceptance Criteria

### Phase 1: Minimal Validation (Month 1)
- [ ] Execute `auralock benchmark-lora` with `--execute` on at least 10 subjects
- [ ] Document GPU setup and training hyperparameters
- [ ] Collect mimicry success rate measurements
- [ ] Publish results in `docs/VALIDATION_RESULTS.md`
- [ ] Calculate correlation: protection_score vs mimicry_prevention
- [ ] Update README to include ground-truth results alongside proxy metrics

### Phase 2: Expanded Validation (Month 2-3)
- [ ] Scale to N=50 subjects
- [ ] Test multiple mimicry methods (DreamBooth, LoRA, Textual Inversion)
- [ ] Compare against academic baselines (Anti-DreamBooth, Mist)
- [ ] Publish validation methodology in academic paper or technical report

### Phase 3: Transparency Upgrades (Ongoing)
- [ ] All benchmark claims must include ground-truth validation status:
  ```
  ✅ Validated: Tested against real LoRA training (N=50, success_rate=32%)
  ⚠️ Proxy only: Not yet validated against real mimicry models
  ```
- [ ] CLI output distinguishes proxy vs ground-truth metrics
- [ ] README never claims protection without validation evidence

### Phase 4: Infrastructure Improvements
- [ ] Add `benchmark_lora_batch.py` script for easy multi-profile validation
- [ ] Create Colab notebook with **executed cells and outputs** (not blank template)
- [ ] Document cost estimation tool: "How much GPU time needed for N subjects?"
- [ ] Provide validation results reproduction guide

## Additional Context

### Why This Hasn't Been Done Yet

Understandable reasons:
1. **GPU costs**: Real validation requires expensive GPU hours
2. **Time**: Training 50 LoRA models takes significant compute time
3. **Complexity**: Managing training runs, collecting outputs, analyzing results

**But**: This is **mandatory for credible benchmark claims**. You cannot claim anti-mimicry protection without measuring actual mimicry prevention.

### What Makes This Critical

From research perspective:
- **Scientific rigor**: You can't publish claims without validation
- **User trust**: Artists need evidence, not proxy scores
- **Development direction**: Need ground-truth feedback to improve methods

From product perspective:
- **False advertising risk**: Claiming protection without validation
- **User harm**: Artists deploy ineffective protection, lose IP
- **Reputation damage**: When protection fails, trust in project collapses

### Recommended Next Action

**Start small**: Run the Phase 1 minimal validation (10 subjects, 5 profiles, DreamBooth only). This costs ~$50 and takes a weekend. Publish honest results, even if protection is weak. **Honesty builds trust more than perfect scores.**

Then iterate based on data.

## References

- `/src/auralock/benchmarks/lora.py` - Complete but unused benchmark harness
- `/src/auralock/benchmarks/antidreambooth.py` - Subject split benchmark infrastructure
- `/notebooks/AuraLock_LoRA_Benchmark_Colab.ipynb` - GPU execution template (no results)
- `/docs/RESEARCH_ROADMAP.md` lines 48-51 - Acknowledges GPU validation is still pending
- README.md lines 52-58 - Unvalidated protection claims
- README.md line 60 - Buried disclaimer about proxy nature
