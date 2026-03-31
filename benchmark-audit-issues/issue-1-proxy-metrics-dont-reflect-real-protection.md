# Issue: Protection Score is a Proxy Metric That May Not Reflect Real Anti-Mimicry Strength

## Labels
`benchmark`, `critical`, `evaluation`, `research`, `technical-debt`

## Problem Description

The repository uses a "Protection Score" derived from style and embedding similarity as the primary metric to evaluate anti-mimicry effectiveness. **This is a proxy metric**, not a ground-truth measurement of actual style mimicry prevention. The score can be high while the protection may still be ineffective against real-world mimicry attacks.

## What is Wrong with the Benchmark

**Current Implementation** (from `/src/auralock/core/metrics.py:308-311`):
```python
protection_score = 100.0 * (
    0.65 * (1.0 - robust_style_similarity)
    + 0.35 * (1.0 - robust_embedding_similarity)
)
```

**Critical Issues**:

1. **Arbitrary Weight Assignment**: The 0.65/0.35 split between style and embedding similarity has no empirical justification. Why not 0.5/0.5? Or 0.8/0.2? These weights are not validated against actual mimicry outcomes.

2. **Single Feature Extractor**: Uses only ResNet18 with ImageNet weights (`/src/auralock/core/style.py:107`). Real-world mimicry models (DreamBooth, LoRA) use completely different architectures (CLIP, VAE, U-Net). The feature space mismatch means:
   - High protection score ≠ actual protection against mimicry models
   - The metric measures "drift in ResNet18 space" not "prevention of style transfer"

3. **No Calibration Against Real Mimicry**: The protection score thresholds (Strong ≥45, Moderate ≥25, Weak <25) are arbitrary. There is no evidence that:
   - A score of 45 actually prevents DreamBooth from learning the style
   - A score of 25 provides any meaningful protection
   - These thresholds correlate with mimicry success/failure rates

4. **Limited Transform Suite** (`/src/auralock/core/style.py:217-231`): Only tests 4 transforms:
   - identity
   - gaussian_blur (kernel=5, sigma=1.0)
   - resize_restore_75
   - resize_restore_50

   Missing many common preprocessing steps that mimicry models use (CLIP preprocessing, VAE encoding, data augmentation).

## Why This Can Mislead Users

### Scenario 1: False Confidence
```python
# User protects artwork, sees "Strong" protection (score=48.2)
# Reality: DreamBooth can still successfully learn the style
# because ResNet18 drift ≠ CLIP/VAE feature preservation
```

### Scenario 2: Unmeasured Attack Surface
Users believe their artwork is protected when:
- The protection score is high in ResNet18 feature space
- But style information remains recoverable in CLIP/VAE space
- Actual mimicry models operate in the unmonitored space

### Scenario 3: Misleading Comparisons
The README claims (lines 52-58):
```
| `fortress` | `53.2` | `29.08` | `0.7858` | more aggressive, visibly harsher output |
| `blindfold` | `61.1` | `26.53` | `0.6114` | strongest current anti-readability preset |
```

**But there's no evidence these profiles actually prevent mimicry.** The scores are relative measurements in a proxy space, not absolute protection guarantees.

## Evidence from Repository

1. **No Ground-Truth Correlation Studies**: Search for validation that protection score predicts mimicry prevention:
   - `/src/tests/test_metrics.py`: Tests metric calculation, not correlation with mimicry success
   - `/src/tests/test_stylecloak.py`: Tests protection score computation, not mimicry prevention
   - **No test validates that high protection scores prevent actual mimicry**

2. **Acknowledged as Proxy** (line 60 in README):
   > "The `Protection Score` is an internal proxy derived from embedding and style similarity after robustness transforms. It is useful for relative comparisons inside this repository, not as a universal guarantee against all AI systems."

   But this critical limitation is buried in documentation, not prominently displayed in results.

3. **Benchmark Harness Exists But No Results**: The repository includes:
   - `/src/auralock/benchmarks/lora.py`: LoRA benchmark infrastructure
   - `/src/auralock/benchmarks/antidreambooth.py`: Anti-DreamBooth benchmark
   - **BUT: No published results correlating protection scores with actual mimicry outcomes**

## Proposed Benchmark Upgrade

### 1. Ground-Truth Validation Study

**Objective**: Measure correlation between protection score and actual mimicry prevention.

**Protocol**:
```
For each profile (safe, balanced, strong, fortress, blindfold):
  1. Protect N=50 diverse artworks
  2. Record protection score for each
  3. Train DreamBooth/LoRA on protected images:
     - Use standard settings (resolution=512, steps=400)
     - Generate M=10 samples per trained model
  4. Measure mimicry success rate:
     - Human evaluation: Does generated image match original style?
     - Automated: CLIP similarity to original style exemplars
  5. Compute correlation: protection_score vs mimicry_success_rate
```

**Expected Output**:
- Scatter plot: protection_score (x-axis) vs mimicry_prevention_rate (y-axis)
- Regression analysis with R² value
- Threshold recalibration based on empirical data

### 2. Multi-Space Feature Evaluation

Replace single ResNet18 with ensemble:
```python
# Current (biased)
extractor = resnet18(weights=ResNet18_Weights.DEFAULT)

# Proposed (comprehensive)
extractors = {
    'resnet18': resnet18(weights=ResNet18_Weights.DEFAULT),
    'clip_vit': CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32'),
    'dino_vit': torch.hub.load('facebookresearch/dino:main', 'dino_vits16'),
}

# Aggregate protection across all spaces
protection_score_ensemble = mean([
    score_from_extractor(extractors[name])
    for name in extractors
])
```

**Rationale**: If protection works in ResNet18 but fails in CLIP space, the overall protection is weak because mimicry models use CLIP.

### 3. Mimicry-Specific Transforms

Extend transform suite to match actual mimicry model preprocessing:
```python
def build_mimicry_aligned_transform_suite():
    return (
        ("identity", identity),
        ("gaussian_blur", gaussian_blur),
        ("resize_restore_75", resize_restore_75),
        ("resize_restore_50", resize_restore_50),
        # NEW: Match actual preprocessing
        ("clip_preprocess", clip_preprocessing_pipeline),
        ("vae_encode_decode", vae_roundtrip),
        ("jpeg_compress_90", jpeg_compression_90),
        ("jpeg_compress_70", jpeg_compression_70),
        ("random_crop_center", center_crop_and_resize),
        ("color_jitter", color_jitter_augment),
    )
```

### 4. Separate Reporting: Proxy vs Ground-Truth

Clearly distinguish in all outputs:
```json
{
  "proxy_metrics": {
    "protection_score_resnet18": 48.2,
    "assessment": "Strong (proxy space)",
    "warning": "This score measures drift in ResNet18 features, NOT actual mimicry prevention"
  },
  "ground_truth_metrics": {
    "mimicry_prevention_rate": null,
    "reason": "Not yet evaluated. Run `auralock benchmark-lora --execute` to validate.",
    "status": "requires_gpu_validation"
  }
}
```

## Acceptance Criteria

### Phase 1: Immediate Fixes (Week 1-2)
- [ ] Add prominent warnings to all protection score outputs:
  - CLI output: "⚠️ Protection score is a proxy metric. Real-world effectiveness not validated."
  - JSON reports: Include `"metric_type": "proxy_unvalidated"` field
  - README: Move warning from line 60 to top of results table (lines 52-58)

### Phase 2: Correlation Study (Month 1-2)
- [ ] Design ground-truth validation protocol (detailed spec document)
- [ ] Implement mimicry success rate measurement tools
- [ ] Run validation study on N=50+ artworks × 5 profiles = 250+ tests
- [ ] Publish correlation analysis: protection_score vs mimicry_prevention_rate
- [ ] Recalibrate thresholds (Strong/Moderate/Weak) based on empirical data

### Phase 3: Multi-Space Protection (Month 2-3)
- [ ] Implement multi-extractor evaluation (ResNet18 + CLIP + DINO)
- [ ] Add mimicry-aligned transforms (CLIP preprocess, VAE, JPEG, crop)
- [ ] Report per-space protection scores and ensemble score
- [ ] Update tests to validate cross-space protection

### Phase 4: Honest Reporting (Ongoing)
- [ ] All benchmark results include both proxy and ground-truth metrics
- [ ] CI/CD checks ensure warnings are present in all reports
- [ ] Documentation clearly distinguishes proxy metrics from validated protection
- [ ] User-facing materials never claim "Strong" protection without ground-truth validation

## Additional Context

This issue is **not about removing the protection score**—it serves a useful purpose for rapid iteration during development. The issue is:
1. **Over-reliance** on the proxy without ground-truth validation
2. **Misleading presentation** that implies the score measures actual protection
3. **Missing correlation studies** to calibrate the proxy against real mimicry

The fix is to:
- **Validate** the proxy against ground-truth outcomes
- **Recalibrate** thresholds based on empirical data
- **Report honestly** about what the metric actually measures
- **Extend** evaluation to feature spaces that mimicry models actually use

## References

- `/src/auralock/core/metrics.py:247-329` - Protection score implementation
- `/src/auralock/core/style.py:217-231` - Limited transform suite
- `/src/auralock/core/style.py:26-101` - Single ResNet18 extractor
- `/src/auralock/benchmarks/lora.py` - Unused ground-truth infrastructure
- README.md lines 52-60 - Unvalidated protection claims
