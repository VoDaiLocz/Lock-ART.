# Issue: Weak Baseline Comparison Methodology - No Comparisons Against Existing Methods

## Labels
`benchmark`, `evaluation`, `research`, `methodology`, `critical`

## Problem Description

The benchmark provides **zero comparisons against existing anti-mimicry methods** or academic baselines. All results are self-referential (comparing AuraLock profiles against each other), with no evaluation against prior art like Anti-DreamBooth, Mist, PhotoGuard, Glaze, or even unprotected baselines. This makes it impossible to assess whether AuraLock represents progress over existing methods or is weaker than state-of-the-art.

## What is Wrong with the Benchmark

### 1. No Baseline Comparisons in Any Results

**README benchmark table** (lines 52-58):
```markdown
| Run | Protection Score | PSNR | SSIM | Notes |
|-----|------------------|------|------|-------|
| `balanced` | `42.1` | `36.24` | `0.9346` | better visual quality, good study baseline |
| `strong` | `48.7` | ... | ... | ... |
| `fortress` | `53.2` | ... | ... | ... |
```

**What's compared**: AuraLock profiles (safe, balanced, strong, fortress, blindfold) against each other.

**What's missing**:
- ❌ **Unprotected baseline**: How do clean images score? (Protection score should be ~0 for unprotected)
- ❌ **Anti-DreamBooth**: Academic SOTA method for preventing DreamBooth mimicry
- ❌ **Mist**: Published adversarial protection method
- ❌ **PhotoGuard**: Another academic approach
- ❌ **Glaze**: Deployed commercial protection system
- ❌ **Simple preprocessing defenses**: JPEG compression, Gaussian noise, blur

**Cannot answer**:
- Is AuraLock better than existing methods?
- Is AuraLock better than doing nothing?
- Is AuraLock better than naive defenses (add noise)?

### 2. Benchmark Infrastructure Doesn't Support External Methods

**Benchmark code** (`/src/auralock/services/protection.py:172-234`):
```python
def benchmark_file(
    image_path: str | Path,
    *,
    profiles: tuple[str, ...] = ("safe", "balanced", "strong"),
    # ...
) -> BenchmarkSummary:
    # Only compares AuraLock's own profiles
    # No way to add external methods like Anti-DreamBooth or Mist
```

**No extension point for external baselines**:
- Can't add "anti_dreambooth" as a profile
- Can't wrap external protection methods
- Can't compare AuraLock vs prior art on same metrics

### 3. Literature Review is Minimal

**Documentation search**:
```bash
$ grep -r "Anti-DreamBooth\|Mist\|Glaze\|PhotoGuard" docs/
# Only mentions in acknowledgements and references, no comparisons
```

**README acknowledgements** (line 243):
```markdown
AuraLock is a learning project shaped by ideas discussed around adversarial artwork protection and anti-mimicry evaluation, especially directions associated with Mist-v2, StyleGuard, Anti-DreamBooth, and related open research.
```

**But**: No actual implementation comparisons, performance comparisons, or evaluation against these methods.

### 4. Anti-DreamBooth Integration But No Comparison

**Infrastructure exists** (`/src/auralock/benchmarks/antidreambooth.py`):
- Implements Anti-DreamBooth-style subject splits (set_A/set_B/set_C)
- Uses Anti-DreamBooth directory structure
- References Anti-DreamBooth scripts

**But** (line 319):
```python
"AuraLock still uses its own protection pipeline; this workflow is a benchmark alignment layer, not an ASPL/FSMG reproduction."
```

**This means**:
- AuraLock **borrows the evaluation protocol** from Anti-DreamBooth
- But **doesn't implement** Anti-DreamBooth's protection method (ASPL/FSMG)
- And **doesn't compare** AuraLock's protection vs Anti-DreamBooth's protection

**Cannot answer**: Is AuraLock better or worse than Anti-DreamBooth?

### 5. No Unprotected Baseline in Reports

**Expected comparison**:
```json
{
  "profile_summaries": {
    "unprotected": {
      "avg_protection_score": 0.0,  // Should be ~0 for clean images
      "avg_psnr_db": null,  // Perfect quality (no perturbation)
      "avg_ssim": 1.0,  // Perfect similarity
    },
    "balanced": {
      "avg_protection_score": 42.1,
      "avg_psnr_db": 36.24,
      "avg_ssim": 0.9346,
    }
  }
}
```

**Actual reports**: No unprotected baseline. Cannot quantify how much protection is gained vs no protection at all.

### 6. No Naive Baseline Comparisons

Simple baseline methods to compare against:
- **Gaussian noise addition**: Add ε-magnitude Gaussian noise to image
- **JPEG compression**: Compress to quality=75, claim "noise makes mimicry harder"
- **Blur**: Apply Gaussian blur, claim "reducing detail prevents mimicry"

**These are naive**, but should be evaluated to show AuraLock is better than trivial approaches.

**Current state**: No comparison, so can't prove AuraLock is better than "just add random noise".

## Why This Can Mislead Users

### Scenario 1: False Impression of Superiority

User reads:
```
| `fortress` | `53.2` | `29.08` | `0.7858` | more aggressive, visibly harsher output |
```

User assumes:
- "Fortress profile is state-of-the-art"
- "53.2 protection score is the best available"
- "This beats all prior methods"

Reality without comparisons:
- Anti-DreamBooth might achieve protection_score of 65.0 (better)
- Or might achieve 30.0 (worse)
- Or protection_score might not be comparable across methods
- **No way to know → User makes uninformed decision**

### Scenario 2: Reinventing the Wheel

Without baseline comparisons:
- AuraLock may be **worse** than existing methods
- Developers waste effort on inferior approach
- Users deploy suboptimal protection
- Community fragments instead of converging on best methods

**Example**: If Anti-DreamBooth achieves 80% mimicry prevention and AuraLock achieves 40%, users should know this.

### Scenario 3: Cannot Assess Cost-Benefit Trade-offs

Different methods have different trade-offs:
```
Method A: High protection, low visual quality
Method B: Medium protection, high visual quality
Method C: Low protection, perfect visual quality (unprotected)
```

**Without comparisons**, users cannot choose the method that best fits their needs:
- Artists prioritizing quality might prefer Method B
- Artists prioritizing protection might prefer Method A
- **But without seeing all options, they can't make informed choice**

### Scenario 4: False Validation

Paper submission:
```
Abstract: "We propose AuraLock, a novel approach to artwork protection."
Results: "AuraLock achieves 53.2 protection score with PSNR=29.08"
```

**Reviewer**: "How does this compare to Anti-DreamBooth [1], Mist [2], and Glaze [3]?"

**Authors**: "We didn't compare against prior art."

**Outcome**: Paper rejected for insufficient evaluation.

**Users who trusted the benchmark are misled by incomplete evaluation.**

## Evidence from Repository

### 1. No Baseline Implementation

Search for baseline methods:
```bash
$ grep -r "class.*Baseline\|def.*baseline" src/
# (no matches)

$ find src/ -name "*baseline*" -o -name "*comparison*"
# (no files)
```

**No baseline comparison infrastructure exists.**

### 2. Profiles Are All AuraLock Variants

**Profile definitions** (`/src/auralock/core/profiles.py`):
```python
PROFILES = {
    "safe": {...},
    "balanced": {...},
    "strong": {...},
    "subject": {...},
    "fortress": {...},
    "blindfold": {...},
}
```

All profiles are AuraLock variants (StyleCloak or Blindfold method). **No external method profiles.**

### 3. Benchmark Code is Closed to External Methods

**Benchmark implementation** (`/src/auralock/services/protection.py:172-234`):
```python
def benchmark_file(..., profiles: tuple[str, ...] = ("safe", "balanced", "strong")):
    results = []
    for profile_name in profiles:
        result = self.protect_file(image_path, profile=profile_name)
        # Only calls self.protect_file() → AuraLock methods only
```

**No plugin system** for external protection methods.

### 4. No Academic Baseline Reproductions

Search for reproduction scripts:
```bash
$ find . -name "*anti_dreambooth*" -o -name "*mist*" -o -name "*glaze*"
./src/auralock/benchmarks/antidreambooth.py  # Uses protocol, not method
# (no implementation of external methods)
```

**Anti-DreamBooth file implements the evaluation protocol (set_A/B/C splits) but not the protection method (ASPL/FSMG).**

### 5. Documentation Acknowledges Prior Work But Doesn't Compare

README line 243 acknowledges:
```
Mist-v2, StyleGuard, Anti-DreamBooth, and related open research.
```

But search for actual comparisons:
```bash
$ grep -A10 -B10 "comparison\|baseline\|vs\|versus" docs/ README.md
# (no comparative evaluation)
```

**Acknowledgement without comparison is insufficient.**

### 6. Test Suite Has No Baseline Tests

Search test files for baseline comparisons:
```bash
$ grep -r "baseline\|comparison" src/tests/
# (no matches)
```

**No tests validate AuraLock performs better than baselines.**

## Proposed Benchmark Upgrade

### Phase 1: Implement Unprotected and Naive Baselines

Create baseline protection methods:
```python
# /src/auralock/baselines/__init__.py

class BaselineProtectionMethod(ABC):
    """Abstract base class for protection methods (AuraLock and baselines)."""

    @abstractmethod
    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply protection to image."""
        pass


class UnprotectedBaseline(BaselineProtectionMethod):
    """Baseline: No protection (identity function)."""

    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        return image


class GaussianNoiseBaseline(BaselineProtectionMethod):
    """Baseline: Add Gaussian noise to image."""

    def __init__(self, epsilon: float = 0.02):
        self.epsilon = epsilon

    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        noise = torch.randn_like(image) * self.epsilon
        return torch.clamp(image + noise, 0.0, 1.0)


class JPEGCompressionBaseline(BaselineProtectionMethod):
    """Baseline: JPEG compression as 'protection'."""

    def __init__(self, quality: int = 75):
        self.quality = quality

    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # JPEG compress and decompress
        from auralock.core.transforms import jpeg_compress_decompress
        return jpeg_compress_decompress(image, quality=self.quality)


class GaussianBlurBaseline(BaselineProtectionMethod):
    """Baseline: Gaussian blur as 'protection'."""

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        from auralock.core.style import gaussian_blur
        return gaussian_blur(image, kernel_size=self.kernel_size, sigma=self.sigma)
```

### Phase 2: Academic Baseline Reproductions

Implement or integrate published methods:

**Option A: Reproduce from Papers**
```python
# /src/auralock/baselines/anti_dreambooth.py

class AntiDreamBoothASPL(BaselineProtectionMethod):
    """Anti-DreamBooth ASPL protection method reproduction."""

    def __init__(self, epsilon: float = 0.05, steps: int = 100):
        # Implement ASPL (Adversarial Style Perturbation Learning)
        # Based on Anti-DreamBooth paper methodology
        ...

    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Apply ASPL protection
        ...
```

**Option B: Wrapper for External Tools** (if they provide APIs):
```python
# /src/auralock/baselines/external_wrappers.py

class GlazeWrapper(BaselineProtectionMethod):
    """Wrapper for Glaze protection tool."""

    def __init__(self, glaze_cli_path: Path):
        self.glaze_cli = glaze_cli_path

    def protect(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Call external Glaze CLI
        # Return protected image
        ...
```

### Phase 3: Unified Benchmark Interface

Extend benchmark to accept any protection method:
```python
# /src/auralock/services/protection.py

def benchmark_file_with_methods(
    image_path: str | Path,
    *,
    methods: dict[str, BaselineProtectionMethod],  # NEW: External methods
    metrics: list[str] = ["psnr", "ssim", "protection_score"],
    ...
) -> BenchmarkSummary:
    """Benchmark multiple protection methods on same image."""

    results = []
    for method_name, method in methods.items():
        # Load image
        image = load_image(image_path)

        # Apply protection
        protected = method.protect(image)

        # Measure metrics
        metrics = compute_all_metrics(image, protected)

        results.append(BenchmarkEntry(
            method=method_name,
            metrics=metrics,
        ))

    return BenchmarkSummary(results=results)
```

Usage:
```python
from auralock.baselines import (
    UnprotectedBaseline,
    GaussianNoiseBaseline,
    AntiDreamBoothASPL,
)

methods = {
    "unprotected": UnprotectedBaseline(),
    "gaussian_noise": GaussianNoiseBaseline(epsilon=0.02),
    "anti_dreambooth": AntiDreamBoothASPL(epsilon=0.05, steps=100),
    "auralock_balanced": AuraLockStyleCloak(profile="balanced"),
    "auralock_fortress": AuraLockStyleCloak(profile="fortress"),
}

summary = benchmark_file_with_methods("artwork.png", methods=methods)
print(summary.comparison_table())
```

### Phase 4: Comparative Benchmark Reports

Generate comparison tables:
```json
{
  "benchmark_type": "comparative",
  "methods_evaluated": [
    "unprotected",
    "gaussian_noise",
    "anti_dreambooth_aspl",
    "auralock_balanced",
    "auralock_fortress"
  ],
  "results": {
    "unprotected": {
      "protection_score": 0.0,
      "psnr_db": null,
      "ssim": 1.0,
      "assessment": "No protection"
    },
    "gaussian_noise": {
      "protection_score": 12.5,
      "psnr_db": 35.2,
      "ssim": 0.92,
      "assessment": "Weak protection, naive baseline"
    },
    "anti_dreambooth_aspl": {
      "protection_score": 48.3,
      "psnr_db": 32.1,
      "ssim": 0.85,
      "assessment": "Strong protection, academic SOTA"
    },
    "auralock_balanced": {
      "protection_score": 42.1,
      "psnr_db": 36.24,
      "ssim": 0.9346,
      "assessment": "Moderate protection, better quality than Anti-DreamBooth"
    },
    "auralock_fortress": {
      "protection_score": 53.2,
      "psnr_db": 29.08,
      "ssim": 0.7858,
      "assessment": "Strong protection, worse quality than Anti-DreamBooth"
    }
  },
  "ranking_by_protection": [
    "auralock_fortress",
    "anti_dreambooth_aspl",
    "auralock_balanced",
    "gaussian_noise",
    "unprotected"
  ],
  "ranking_by_quality": [
    "unprotected",
    "auralock_balanced",
    "gaussian_noise",
    "anti_dreambooth_aspl",
    "auralock_fortress"
  ],
  "pareto_frontier": [
    "auralock_balanced",  // Best quality-protection trade-off
    "anti_dreambooth_aspl"  // Strong protection, acceptable quality
  ]
}
```

### Phase 5: Ground-Truth Comparative Validation

Run LoRA/DreamBooth training on all methods:
```yaml
validation_protocol:
  dataset: 50 diverse artworks
  protection_methods:
    - unprotected (baseline)
    - gaussian_noise (naive baseline)
    - anti_dreambooth_aspl (academic baseline)
    - auralock_balanced
    - auralock_fortress

  mimicry_evaluation:
    model: DreamBooth LoRA
    training_steps: 400
    samples_per_model: 20

  metrics:
    - mimicry_success_rate (primary)
    - protection_score (proxy)
    - PSNR, SSIM (quality)
    - training_time, compute_cost

  expected_output:
    # Table: method → mimicry_prevention_rate
    # Ranking by actual protection (ground truth)
    # Correlation: proxy_score vs actual_prevention
```

### Phase 6: Update README with Honest Comparisons

Replace current table with comparative results:
```markdown
## Comparative Benchmark Results

Comparison of AuraLock profiles against baselines and academic methods.

### Proxy Metrics (ResNet18 Feature Space)

| Method | Protection Score | PSNR | SSIM | Quality-Protection Trade-off |
|--------|------------------|------|------|------------------------------|
| Unprotected | 0.0 | ∞ | 1.0 | Perfect quality, no protection |
| Gaussian Noise (ε=0.02) | 12.5 | 35.2 | 0.92 | Weak protection, naive baseline |
| Anti-DreamBooth ASPL | 48.3 | 32.1 | 0.85 | Strong protection, academic SOTA |
| **AuraLock Balanced** | **42.1** | **36.24** | **0.9346** | **Better quality than ASPL, moderate protection** |
| **AuraLock Fortress** | **53.2** | **29.08** | **0.7858** | **Strongest protection, lower quality** |

### Ground-Truth Mimicry Prevention (DreamBooth LoRA)

| Method | Mimicry Success Rate | Protection Effectiveness | Visual Quality |
|--------|----------------------|-------------------------|----------------|
| Unprotected | 95% (baseline) | 0% prevented | Perfect |
| Gaussian Noise | 87% | 8% prevented ⚠️ | Good |
| Anti-DreamBooth ASPL | 42% | 53% prevented ✓ | Acceptable |
| **AuraLock Balanced** | **62%** | **33% prevented** ⚠️ | **Good** |
| **AuraLock Fortress** | **48%** | **47% prevented** ✓ | **Acceptable** |

**Key Findings**:
- AuraLock Balanced offers better visual quality than Anti-DreamBooth but weaker protection
- AuraLock Fortress is competitive with Anti-DreamBooth SOTA
- Gaussian noise is ineffective (only 8% prevention)
- Correlation: protection_score ≈ mimicry_prevention (R²=0.78)

**Trade-off Recommendation**:
- Prioritize quality: Use AuraLock Balanced
- Prioritize protection: Use Anti-DreamBooth ASPL or AuraLock Fortress
- Naive baselines (Gaussian noise) are not recommended
```

## Acceptance Criteria

### Phase 1: Naive Baselines (Week 1)
- [ ] Implement `UnprotectedBaseline` class
- [ ] Implement `GaussianNoiseBaseline` class
- [ ] Implement `JPEGCompressionBaseline` class
- [ ] Implement `GaussianBlurBaseline` class
- [ ] Add tests comparing AuraLock vs naive baselines

### Phase 2: Academic Baseline Reproduction (Week 2-4)
- [ ] Reproduce Anti-DreamBooth ASPL method
- [ ] Validate reproduction matches paper results
- [ ] Integrate into benchmark framework
- [ ] Add Mist baseline (if feasible)
- [ ] Add PhotoGuard baseline (if feasible)

### Phase 3: Unified Benchmark Interface (Week 3)
- [ ] Implement `BaselineProtectionMethod` abstract class
- [ ] Refactor AuraLock methods to inherit from base class
- [ ] Implement `benchmark_file_with_methods()` function
- [ ] Add CLI support: `--methods unprotected,gaussian_noise,anti_dreambooth,balanced`

### Phase 4: Comparative Results (Week 4-5)
- [ ] Run comparative benchmark on diverse dataset (N≥50 images)
- [ ] Generate comparison tables and plots
- [ ] Identify Pareto frontier (best quality-protection trade-offs)
- [ ] Update README with comparative results

### Phase 5: Ground-Truth Validation (Month 2-3)
- [ ] Run LoRA/DreamBooth training on all methods
- [ ] Measure actual mimicry prevention rates
- [ ] Compare proxy scores vs ground-truth prevention
- [ ] Publish validation results

### Phase 6: Honest Documentation (Ongoing)
- [ ] All benchmark claims include baseline comparisons
- [ ] README clearly states AuraLock's position vs SOTA
- [ ] Acknowledge where baselines perform better
- [ ] Provide trade-off guidance for users

## Additional Context

### Why Baselines Matter

From scientific methodology:
- **Internal validity**: Does your method work as intended?
- **External validity**: Does it work better than alternatives?

**Without baselines**: Only internal validity. Cannot claim your method is useful.

### Pareto Optimality

Different methods may excel in different dimensions:
```
Method A: High protection, low quality
Method B: Medium protection, medium quality
Method C: Low protection, high quality
```

If Method B is **Pareto dominated** (worse than A in protection AND worse than C in quality), it should not be recommended.

**Need baselines to identify Pareto-optimal methods.**

### Academic Standards

Conference reviews expect:
- Comparison against prior SOTA
- Ablation studies
- Statistical significance testing
- Honest assessment of limitations

**Current benchmark would not pass peer review without baseline comparisons.**

## References

- README.md lines 52-58 - No baseline comparisons in results
- `/src/auralock/core/profiles.py` - Only AuraLock variants
- `/src/auralock/benchmarks/antidreambooth.py:319` - Uses protocol but not method
- README line 243 - Acknowledges prior work without comparing
- Anti-DreamBooth paper: [arXiv:2303.15433](https://arxiv.org/abs/2303.15433)
- Mist paper: [arXiv:2305.01894](https://arxiv.org/abs/2305.01894)
