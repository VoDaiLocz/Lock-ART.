# Benchmark Audit Issues - Summary Index

This directory contains a comprehensive technical audit of the Lock-ART/AuraLock benchmark methodology. Each issue identifies specific weaknesses in benchmark integrity and anti-mimicry evaluation quality.

## Overview

The audit was conducted on 2026-03-31 and identified **7 critical benchmark issues** that can mislead users about protection effectiveness.

## Critical Issues

### Issue 1: Proxy Metrics Don't Reflect Real Anti-Mimicry Strength
**File**: `issue-1-proxy-metrics-dont-reflect-real-protection.md`

**Summary**: Protection Score is a proxy metric (ResNet18 feature drift) that may not correlate with actual mimicry prevention. Arbitrary weight assignment (0.65/0.35), single feature extractor, no calibration against real mimicry outcomes.

**Impact**: HIGH - Users may trust high protection scores that don't reflect real-world effectiveness.

**Key Problems**:
- Arbitrary 0.65/0.35 weight split with no empirical justification
- Single ResNet18 extractor vs actual CLIP/VAE in mimicry models
- No validation that high scores prevent actual mimicry
- Thresholds (Strong ≥45, Moderate ≥25) are uncalibrated

---

### Issue 2: Missing Ground-Truth LoRA/DreamBooth Validation
**File**: `issue-2-missing-ground-truth-validation.md`

**Summary**: Complete benchmark infrastructure exists (LoRA harness, Anti-DreamBooth splits, Docker runtime) but **zero published validation results**. All protection claims based on proxy metrics without demonstrating actual mimicry prevention.

**Impact**: CRITICAL - Cannot verify if protection actually works against real mimicry models.

**Key Problems**:
- Infrastructure present but never executed with `execute=True` for published results
- No evidence that protection survives DreamBooth/LoRA training
- Colab notebook exists but has no executed outputs
- Roadmap acknowledges GPU validation is "pending" (still not done)

---

### Issue 3: Weak Robustness Testing
**File**: `issue-3-weak-robustness-testing.md`

**Summary**: Only 4 basic transforms tested (identity, blur, 2×resize). Missing critical preprocessing: JPEG compression, center crop, CLIP preprocessing, VAE encoding, color jitter, rotation. Protection may survive blur but fail against JPEG.

**Impact**: HIGH - False robustness claims. Purification attacks not tested.

**Key Problems**:
- No JPEG compression (most effective purification defense)
- Resize-restore ≠ JPEG (different artifacts)
- No crop testing (standard augmentation)
- No CLIP/VAE preprocessing (actual mimicry pipeline)
- Single blur configuration (kernel=5, sigma=1.0)

---

### Issue 4: Dataset Split Methodology Risks
**File**: `issue-4-dataset-split-methodology-risks.md`

**Summary**: No train/val/test split separation or enforcement. Risk of overfitting, data leakage, and cherry-picking results. Anti-DreamBooth set_C is "metadata only", not used for held-out evaluation.

**Impact**: HIGH - Results may be overfit to test set. Cannot verify generalization.

**Key Problems**:
- Local benchmark has no split methodology at all
- Nothing prevents tuning on test images
- set_C described as "holdout metadata" (not actually held out)
- No split tracking in reports
- Cherry-picking not prevented

---

### Issue 5: Insufficient Reproducibility Documentation
**File**: `issue-5-insufficient-reproducibility-documentation.md`

**Summary**: Cannot reproduce or verify published results. Missing: exact datasets, model versions, random seeds, hardware specs, environment details, execution logs. Published results exist only in README markdown without archived data.

**Impact**: HIGH - Cannot independently verify benchmark claims. Violates scientific standards.

**Key Problems**:
- No benchmark result archives (no `benchmark_results/` directory)
- Reports missing environment metadata (PyTorch version, hardware, timestamp)
- Loose version constraints (torch>=2.0.0 allows drift)
- No reproducibility guide
- README results have no source trace

---

### Issue 6: Weak Baseline Comparison Methodology
**File**: `issue-6-weak-baseline-comparison-methodology.md`

**Summary**: Zero comparisons against existing methods. All results are self-referential (AuraLock profiles vs each other). No evaluation against Anti-DreamBooth, Mist, Glaze, PhotoGuard, or even unprotected baselines.

**Impact**: CRITICAL - Cannot assess if AuraLock represents progress or is weaker than SOTA.

**Key Problems**:
- No unprotected baseline (can't quantify protection gained)
- No naive baselines (Gaussian noise, JPEG, blur)
- No academic baselines (Anti-DreamBooth, Mist)
- Anti-DreamBooth infrastructure uses protocol but not method
- Benchmark closed to external methods (no plugin system)

---

### Issue 7: Misleading Benchmark Framing in README
**File**: `issue-7-misleading-benchmark-framing.md`

**Summary**: Unvalidated claims presented as facts using definitive language. Critical limitations buried in fine print after impressive numbers. Claims "honest evaluation" while using marketing presentation patterns.

**Impact**: CRITICAL - Misleads users into false security. Contradicts "honest" framing claim.

**Key Problems**:
- Definitive results (lines 52-58) before disclaimer (line 60)
- Profile names suggest validation ("fortress", "blindfold")
- "Benchmark" terminology implies validation
- "Honest evaluation" claim contradicted by framing
- Repository description implies proven "cloaking"

---

## Severity Assessment

| Issue | Severity | User Impact | Implementation Effort |
|-------|----------|-------------|----------------------|
| Issue 1: Proxy Metrics | HIGH | False confidence in protection | Medium (2-3 months for full fix) |
| Issue 2: No Ground-Truth | CRITICAL | Cannot verify protection works | High (GPU validation needed) |
| Issue 3: Weak Robustness | HIGH | Purification attacks undetected | Medium (2-4 weeks) |
| Issue 4: Dataset Splits | HIGH | Overfitting risk | Medium (2-3 weeks) |
| Issue 5: Reproducibility | HIGH | Cannot verify claims | Medium (2-3 weeks) |
| Issue 6: No Baselines | CRITICAL | Cannot assess vs SOTA | High (need baseline reproductions) |
| Issue 7: Misleading Framing | CRITICAL | User deception risk | Low (1-2 weeks documentation) |

## Audit Methodology

### Audit Scope
- Repository: `VoDaiLocz/Lock-ART.` (github.com)
- Commit: `703230e` (Initial plan, 2026-03-31)
- Branch: `claude/audit-benchmark-issues`
- Focus: Benchmark integrity and anti-mimicry evaluation quality

### Audit Process
1. Explored repository structure and benchmark code
2. Analyzed evaluation methodology and metrics
3. Reviewed documentation and published claims
4. Identified gaps between claims and validation
5. Assessed user deception risk
6. Proposed technical fixes with acceptance criteria

### Audit Standards
- **Strict technical rigor**: No leniency for unvalidated claims
- **User protection focus**: Prioritize preventing false security
- **Scientific standards**: Require reproducibility and validation
- **Honest communication**: Demand prominent disclaimers

## Recommendations

### Immediate Actions (Week 1)
1. **Add prominent disclaimers** to README (Issue #7)
   - Validation status section at top
   - "Proxy only, unvalidated" labels on all results
   - CLI warnings when protection is used

2. **Archive current results** with full metadata (Issue #5)
   - Create `benchmark_results/` directory
   - Capture environment info
   - Generate reproduction scripts

3. **Implement JPEG compression testing** (Issue #3)
   - Most critical robustness gap
   - Can be done quickly without GPU

### Short-Term Priorities (Month 1)
1. **Run minimal ground-truth validation** (Issue #2)
   - 10 subjects × 5 profiles = 50 LoRA training runs
   - Cost: ~$50 on Google Colab Pro
   - Publish honest results even if protection is weak

2. **Add unprotected and naive baselines** (Issue #6)
   - Unprotected (identity)
   - Gaussian noise
   - JPEG compression
   - Gaussian blur

3. **Implement dataset split methodology** (Issue #4)
   - Create train/val/test splits
   - Add split tracking to reports
   - Use set_C for held-out validation

### Medium-Term Goals (Months 2-3)
1. **Comprehensive ground-truth validation** (Issue #2)
   - Scale to 50+ subjects
   - Multiple mimicry methods (DreamBooth, LoRA, Textual Inversion)
   - Statistical significance testing

2. **Reproduce academic baselines** (Issue #6)
   - Anti-DreamBooth ASPL method
   - Mist (if reproducible)
   - Comparative evaluation

3. **Multi-space feature evaluation** (Issue #1)
   - Add CLIP and DINO extractors
   - Evaluate protection in multiple feature spaces
   - Correlate proxy scores with mimicry prevention

### Long-Term Vision (Months 3-6)
1. **Establish rigorous benchmark standard**
   - Ground-truth validation as requirement
   - Baseline comparisons mandatory
   - Reproducibility infrastructure automated

2. **Independent verification**
   - Invite external researchers to validate
   - Publish results in peer-reviewed venue
   - ACM artifact evaluation badges

3. **Continuous validation CI/CD**
   - Monthly GPU validation runs
   - Alert on protection degradation
   - Public results dashboard

## Expected Outcomes

### If Protection Works Well
- Publish validation evidence
- Update claims with confidence
- Establish as credible SOTA baseline
- Contribute to research community

### If Protection Works Poorly
- Report honest findings
- Identify failure modes
- Guide research improvements
- Build trust through transparency

### Either Way
- **Honesty builds trust** more than perfect scores
- Users deserve truth, not marketing
- Science requires validation, not claims
- Iterate based on evidence, not assumptions

## Contact

This audit was conducted as a technical review of benchmark methodology. For questions or discussion:

- Open issues on the repository
- Reference specific issue files in discussions
- Follow acceptance criteria for fixes

## License

This audit documentation is provided for the benefit of the Lock-ART project and research community. Issues should be addressed systematically according to severity and user impact.

---

**Audit Date**: 2026-03-31
**Audit Version**: 1.0
**Repository**: VoDaiLocz/Lock-ART. (github.com)
**Branch**: claude/audit-benchmark-issues
