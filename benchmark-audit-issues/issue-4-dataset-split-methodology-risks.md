# Issue: Dataset Split Methodology Risks - Missing Train/Val/Test Separation

## Labels
`benchmark`, `data-leakage`, `methodology`, `research`, `technical-debt`

## Problem Description

The repository lacks clear separation between training/validation/test splits, creating risk of **data leakage**, **overfitting to test sets**, and **cherry-picking results**. The Anti-DreamBooth subject split (set_A/set_B/set_C) exists but its usage is not enforced or validated, and the local benchmark workflow has no split separation at all.

## What is Wrong with the Benchmark

### 1. Local Benchmark Has No Split Methodology

**Current workflow** (`auralock benchmark`):
```python
# /src/auralock/services/protection.py:172-234
def benchmark_file(image_path, profiles, ...):
    # Apply each profile to the SAME image
    # Report metrics on the SAME image
    # No train/validation/test split
```

**Problem**: If you tune profiles based on benchmark results, you're **training on your test set**.

Example workflow that causes overfitting:
```bash
# Step 1: Benchmark current profiles
$ auralock benchmark artwork.png --profiles safe,balanced,strong

# Step 2: See that "safe" has low protection score
# Step 3: Tune "safe" profile parameters to increase score on artwork.png
# Step 4: Benchmark again on artwork.png
# Step 5: Claim improvement

# Problem: You overfitted to artwork.png. No generalization guarantee.
```

### 2. Anti-DreamBooth Split Usage Not Validated

**Infrastructure exists** (`/src/auralock/benchmarks/antidreambooth.py:110-149`):
```python
def resolve_subject_layout(subject_root: str | Path):
    # Validates set_A, set_B, set_C exist
    # Returns layout with image paths for each split

    # set_A: Reference/tuning split (clean)
    # set_B: Published/training split (protection target)
    # set_C: Holdout validation split
```

**But no enforcement**:
- Nothing prevents training on set_C (holdout)
- Nothing validates that set_A/B/C are truly independent
- No checks for duplicate images across splits
- No documentation on split ratios or methodology

**From code** (`/src/auralock/benchmarks/antidreambooth.py:316-319`):
```python
notes = [
    "set_A is retained as a clean reference split, set_B is treated as the published split, and set_C is preserved as holdout metadata.",
]
```

"Preserved as holdout **metadata**" suggests set_C is not being used for held-out evaluation—just kept as metadata. **This defeats the purpose of a holdout set.**

### 3. No Protection Against Cherry-Picking

**Nothing prevents**:
```bash
# Run benchmark on 100 artworks
$ for img in artworks/*.png; do
    auralock benchmark $img --profiles balanced --report reports/${img}.json
done

# Cherry-pick the 10 best results
$ python select_best_results.py --top 10

# Publish only the cherry-picked results in README
```

Users see impressive protection scores, unaware that:
- 90 artworks had poor protection (not shown)
- Results are not representative of typical performance
- Selection bias inflates reported effectiveness

### 4. Collective Protection Without Split Validation

**Collective mode** (`/src/auralock/services/protection.py:296-385`):
```python
def _protect_directory_collectively(...):
    # Compute batch-level perturbation
    # Share 30% of perturbation across all images
    # Ensure consistency across subject sets
```

**Problem**: If you benchmark collective protection on the same set you protected, you're measuring **in-sample** performance. No guarantee of **out-of-sample** effectiveness.

**Correct methodology**:
1. Apply collective protection to training split (set_B)
2. Measure protection score on held-out validation split (set_C)
3. Report out-of-sample metrics

**Current practice**: No evidence this is being done.

### 5. Benchmark Reports Don't Track Data Lineage

**Report structure** (`/src/auralock/services/protection.py:19-72`):
```python
@dataclass
class BenchmarkSummary:
    input_path: Path
    image_count: int
    entries: list[BenchmarkEntry]
    profile_summaries: dict[str, ProfileAggregate]

    # Missing:
    # - split: "train" | "val" | "test"
    # - split_metadata: hash of split assignment
    # - parent_dataset: name of dataset split belongs to
```

Without tracking split membership, you can't:
- Verify results are from held-out test sets
- Reproduce split assignments
- Audit for data leakage

## Why This Can Mislead Users

### Scenario 1: Overfitting to Benchmark Images

Developer workflow:
```python
# Week 1: Benchmark on portrait.png
balanced_score = 42.1  # Low protection

# Week 2: Tune balanced profile to maximize protection on portrait.png
# Increase epsilon from 0.02 to 0.025, adjust alpha, increase steps

# Week 3: Benchmark again on portrait.png
balanced_score = 48.7  # "Improvement"!

# Publish claim: "We improved protection by 15.7%"
```

**Problem**: Improvement is specific to portrait.png. May not generalize to other artworks. **No held-out validation performed.**

### Scenario 2: Cherry-Picking Favorable Results

Marketing workflow:
```python
# Benchmark 100 diverse artworks
results = []
for artwork in diverse_dataset:
    result = benchmark(artwork, profile="fortress")
    results.append(result)

# Select top 5 results where protection_score > 55
best_results = [r for r in results if r.protection_score > 55][:5]

# Publish only best_results in README
# README table shows: fortress achieves 55-62 protection scores
```

**Users don't know**:
- 95% of artworks had lower protection (30-45 range)
- Results are cherry-picked outliers
- Typical performance is much worse than advertised

### Scenario 3: Training on Test Set

Research workflow:
```python
# Collect dataset of 50 artworks
# Run comprehensive benchmark to find best hyperparameters
# Iterate 20 times, adjusting parameters based on benchmark results
# Publish final "best" configuration

# Problem: All 50 artworks have been seen during tuning
# "Best" configuration is overfit to this specific dataset
# No held-out test set to validate generalization
```

When users apply the "best" profile to their artwork:
- Performance may be significantly worse
- Published benchmarks were overfit to development set
- **No true test set evaluation was performed**

## Evidence from Repository

### 1. No Split Enforcement in Benchmarking Code

Search for train/val/test split logic:
```bash
$ grep -r "train.*split\|val.*split\|test.*split" src/
# (no matches)

$ grep -r "holdout" src/
# Only in antidreambooth.py as comment, not enforced
```

**No code enforces split separation.**

### 2. Benchmark Reports Don't Track Splits

Examine report dataclasses:
```python
# /src/auralock/services/protection.py:19-72
@dataclass
class BenchmarkSummary:
    # ... fields ...
    # No "split" field
    # No "split_hash" field
    # No validation that images are from held-out set
```

**Reports don't document which split was used.**

### 3. Anti-DreamBooth Split is "Metadata" Only

From `/src/auralock/benchmarks/antidreambooth.py:318`:
```python
"set_C is preserved as holdout metadata."
```

"Metadata" suggests set_C is not being actively used for held-out evaluation. **This is a red flag.**

### 4. README Benchmark Results Have No Split Info

README lines 52-58:
```
| Run | Protection Score | PSNR | SSIM | Notes |
|-----|------------------|------|------|-------|
| `balanced` | `42.1` | `36.24` | `0.9346` | better visual quality, good study baseline |
```

**Missing information**:
- Which images were these scores measured on?
- Were images part of development/tuning process?
- Are these in-sample or out-of-sample results?
- What dataset? How many images?

**Cannot verify if results are from held-out test set.**

### 5. Documentation Acknowledges Need But Doesn't Implement

`/docs/system-design/10_BENCHMARK_DESIGN.md:20-22`:
```markdown
## Anti-bias
- Không cherry-pick ảnh đẹp nhất.
- Tách tập tune/test.
```

Translation: "Don't cherry-pick best images. Separate tune/test sets."

**This is acknowledged as needed but not implemented or enforced.**

## Proposed Benchmark Upgrade

### Phase 1: Formalize Split Methodology

Create `SplitMetadata` dataclass:
```python
from dataclasses import dataclass
from enum import Enum

class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"
    DEVELOPMENT = "dev"  # Used for tuning, not final eval

@dataclass
class SplitMetadata:
    split_type: SplitType
    dataset_name: str
    dataset_version: str
    split_hash: str  # Deterministic hash of split assignment
    image_ids: list[str]  # Unique IDs of images in this split
    split_method: str  # "random", "stratified", "subject_based"
    split_ratio: dict[str, float]  # {"train": 0.7, "val": 0.15, "test": 0.15}
    random_seed: int | None  # For reproducibility

    def verify_no_leakage(self, other: 'SplitMetadata') -> bool:
        """Check that no images overlap between splits."""
        return set(self.image_ids).isdisjoint(set(other.image_ids))
```

### Phase 2: Enforce Split Separation in Benchmarking

Update benchmark functions to require split metadata:
```python
# /src/auralock/services/protection.py

def benchmark_file(
    image_path: str | Path,
    *,
    split_metadata: SplitMetadata,  # NEW: Required
    profiles: tuple[str, ...] = ("safe", "balanced", "strong"),
    ...
) -> BenchmarkSummary:
    """Benchmark profiles on a file with split tracking."""

    # Validate image belongs to correct split
    if str(image_path) not in split_metadata.image_ids:
        raise ValueError(
            f"Image {image_path} not found in declared split {split_metadata.split_type}. "
            "Potential data leakage."
        )

    # Enforce: only TEST split allowed for final benchmark reporting
    if split_metadata.split_type != SplitType.TEST:
        warnings.warn(
            f"Benchmarking on {split_metadata.split_type} split. "
            "Results may be overfit. Use TEST split for final evaluation."
        )

    # ... existing benchmark logic ...

    # Add split metadata to report
    summary.split_metadata = split_metadata
    return summary
```

### Phase 3: Implement Dataset Split Utilities

Create `auralock.benchmarks.splits` module:
```python
def create_random_split(
    image_paths: list[Path],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[SplitType, SplitMetadata]:
    """Create random train/val/test split with metadata."""

    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    random.seed(random_seed)
    shuffled = random.sample(image_paths, len(image_paths))

    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)

    train_images = shuffled[:n_train]
    val_images = shuffled[n_train:n_train + n_val]
    test_images = shuffled[n_train + n_val:]

    return {
        SplitType.TRAIN: SplitMetadata(
            split_type=SplitType.TRAIN,
            image_ids=[str(p) for p in train_images],
            split_method="random",
            random_seed=random_seed,
            # ... other fields ...
        ),
        # ... VAL and TEST ...
    }


def save_split_manifest(splits: dict[SplitType, SplitMetadata], output_path: Path):
    """Save split assignments for reproducibility."""
    manifest = {
        split_type.value: split_meta.to_dict()
        for split_type, split_meta in splits.items()
    }
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_split_manifest(manifest_path: Path) -> dict[SplitType, SplitMetadata]:
    """Load previously saved split assignments."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {
        SplitType(key): SplitMetadata.from_dict(value)
        for key, value in manifest.items()
    }
```

### Phase 4: CLI Commands for Split Management

```bash
# Create and save split manifest
$ auralock split create ./dataset --output splits.json \
    --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
    --seed 42

# Validate no leakage
$ auralock split validate splits.json

# Benchmark only on test split
$ auralock benchmark ./dataset \
    --split-manifest splits.json \
    --split-type test \
    --profiles balanced,strong \
    --report test_results.json

# Attempt to benchmark on train split -> warning
$ auralock benchmark ./dataset \
    --split-manifest splits.json \
    --split-type train \
    --profiles balanced

⚠️  WARNING: Benchmarking on TRAIN split. Results may be overfit.
             Use --split-type test for final evaluation.
```

### Phase 5: Validation Checks in CI/CD

Add split validation to CI:
```yaml
# .github/workflows/benchmark-validation.yml
- name: Validate Split Methodology
  run: |
    # Check that published benchmark results are from TEST split
    python scripts/validate_published_results.py

    # Verify no data leakage
    python scripts/check_split_leakage.py --splits splits.json

    # Ensure benchmark reports include split metadata
    python scripts/audit_benchmark_reports.py --reports reports/
```

### Phase 6: Update Anti-DreamBooth to Use set_C Correctly

**Current**: set_C is "metadata"

**Proposed**: Use set_C for held-out evaluation:
```python
# /src/auralock/benchmarks/antidreambooth.py

class AntiDreamBoothSubjectBenchmarkHarness:
    def run_with_holdout_validation(self, ...):
        # Step 1: Protect set_B (published split)
        self._prepare_protected_split(layout.set_b_dir, ...)

        # Step 2: Train on protected set_B
        train_on_protected_set_B(...)

        # Step 3: Evaluate mimicry success on held-out set_C
        holdout_metrics = evaluate_mimicry_on_holdout(
            trained_model,
            holdout_images=layout.set_c_images,
        )

        # Step 4: Report out-of-sample mimicry prevention
        return {
            "protected_training_set": "set_B",
            "holdout_evaluation_set": "set_C",
            "holdout_mimicry_success_rate": holdout_metrics.success_rate,
            "holdout_protection_score": holdout_metrics.protection_score,
        }
```

## Acceptance Criteria

### Phase 1: Split Metadata Infrastructure (Week 1)
- [ ] Implement `SplitMetadata` dataclass
- [ ] Implement `create_random_split()` utility
- [ ] Implement split manifest save/load functions
- [ ] Add tests for split creation and validation

### Phase 2: Enforce Split Separation (Week 2)
- [ ] Update `benchmark_file()` to require and validate split metadata
- [ ] Update `benchmark_directory()` to require split manifest
- [ ] Add warnings when benchmarking on non-test splits
- [ ] Raise errors on data leakage detection

### Phase 3: CLI and Documentation (Week 2-3)
- [ ] Add `auralock split create` command
- [ ] Add `auralock split validate` command
- [ ] Update benchmark commands to accept `--split-manifest` and `--split-type`
- [ ] Document split methodology in README and system design docs

### Phase 4: Audit Published Results (Week 3)
- [ ] Audit all published benchmark results in README
- [ ] Add split metadata to result tables
- [ ] Mark results as "in-sample" or "out-of-sample"
- [ ] Re-run benchmarks on proper held-out test sets if needed

### Phase 5: CI/CD Validation (Week 4)
- [ ] Add split validation checks to CI
- [ ] Prevent merging benchmark results without split metadata
- [ ] Automated leakage detection in pull requests

### Phase 6: Anti-DreamBooth Holdout Evaluation (Week 4-5)
- [ ] Update Anti-DreamBooth harness to use set_C for held-out validation
- [ ] Measure mimicry success on held-out images
- [ ] Report out-of-sample protection effectiveness

## Additional Context

### Why Split Methodology Matters

From machine learning best practices:
1. **Training set**: Data used to train/tune model (profile parameters)
2. **Validation set**: Data used to select best hyperparameters
3. **Test set**: **Never touched until final evaluation**

If you tune on test set:
- **Overfitting**: Parameters optimized for specific test images
- **Optimistic bias**: Performance estimates are inflated
- **Generalization failure**: Performance collapses on new data

### Analogous Problem in Adversarial Robustness

CVPR 2019 exposed "adaptive attack" problem:
- Defenses were evaluated on **non-adaptive attacks**
- When attacks were adapted to defense, performance collapsed
- Many "robust" defenses were actually weak

**Similar risk here**:
- Protection tuned on benchmark images
- Attacker adapts to protection method
- Protection effectiveness collapses on real data

### Dataset Size Considerations

Current approach uses **small datasets** (10-50 images):
- Risk of overfitting is **very high**
- Even worse if same images used for tuning and evaluation
- Need much larger held-out test sets for reliable estimates

**Recommendation**: Minimum 100 images per split (300 total) for credible benchmarking.

## References

- `/src/auralock/services/protection.py:172-234` - Benchmark without split tracking
- `/src/auralock/benchmarks/antidreambooth.py:316-319` - set_C as "metadata" only
- `/docs/system-design/10_BENCHMARK_DESIGN.md:20-22` - Acknowledges need for split separation
- Machine learning evaluation methodology (e.g., [Hastie et al. ESL Ch. 7](https://web.stanford.edu/~hastie/ElemStatLearn/))
- CVPR 2019 Adversarial Robustness Workshop on adaptive attacks
