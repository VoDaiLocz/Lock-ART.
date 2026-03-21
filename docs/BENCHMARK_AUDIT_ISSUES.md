# Benchmark Integrity Audit - GitHub Issue Drafts

The following are strict, technical, ready-to-file issue drafts for benchmark integrity and anti-mimicry evaluation quality.

---

## 1) Proxy protection score is overfit to internal feature extractor

**Title**
`Benchmark: protection_score is an internal proxy, not evidence of anti-mimicry robustness`

**What is wrong with the benchmark**
The main benchmark metric (`protection_score`) is computed from internal style/embedding similarities produced by AuraLock’s own feature stack and transform suite. It does not measure downstream anti-mimicry failure on external training pipelines.

**Why it can mislead users**
Users can interpret high `protection_score` as “strong anti-mimicry,” while the metric only reflects divergence under one internal feature representation. This risks false confidence when deployed against real LoRA/DreamBooth workflows.

**Evidence from the repo**
- `protection_score` is explicitly derived from internal robust style/embedding similarity averages: `src/auralock/core/metrics.py:287-311`
- Style/embedding features are produced via a local ResNet18-based extractor: `src/auralock/core/style.py:26-117`
- README states this is an “internal proxy” and not universal: `README.md:60`

**Proposed benchmark upgrade**
Add outcome-grounded metrics alongside `protection_score`:
- Subject fidelity drop vs clean baseline after LoRA/DreamBooth training (e.g., CLIP-I/T, identity similarity, prompt adherence).
- Attack success on held-out prompts/seeds after training.
- Report proxy metrics as secondary diagnostics only.

**Acceptance criteria**
- [ ] Report schema includes at least one downstream model-outcome metric per run (not only proxy similarity).
- [ ] Benchmark table separates **proxy** vs **ground-truth outcome** sections.
- [ ] CI test fails if benchmark report omits the ground-truth outcome section.

**Labels**
`benchmark`, `metrics`, `anti-mimicry`, `research-quality`, `needs-validation`

---

## 2) Benchmark framing overstates evaluation completeness

**Title**
`Benchmark framing: CLI/README present profile benchmark as completion despite missing external attack outcomes`

**What is wrong with the benchmark**
`auralock benchmark` summarizes averages of PSNR/SSIM/protection_score/runtime across profiles and prints “Benchmark completed,” but does not evaluate any model training or mimicry outcomes.

**Why it can mislead users**
The command name/output implies anti-mimicry performance benchmarking, but it is only an internal perturbation-quality/profile comparison. This can be read as stronger evidence than it is.

**Evidence from the repo**
- CLI benchmark only calls `benchmark_file/benchmark_directory` and renders profile averages: `src/auralock/cli.py:590-651`
- Summary fields are only PSNR/SSIM/protection_score/runtime averages: `src/auralock/services/protection.py:718-742`
- README “Current Study Snapshot” highlights protection score rows without downstream attack outcomes: `README.md:48-60`

**Proposed benchmark upgrade**
Split CLI and docs into two explicit benchmark tiers:
1. `benchmark-proxy` (current behavior)
2. `benchmark-attack` (real LoRA/DreamBooth execution + outcome metrics)

Use explicit warning banners when only proxy tier is run.

**Acceptance criteria**
- [ ] Proxy benchmark command/output includes “proxy-only” wording in command help and report JSON.
- [ ] Attack benchmark command exists and outputs model-outcome metrics.
- [ ] README tables distinguish proxy runs from true attack-evaluated runs.

**Labels**
`benchmark`, `ux`, `documentation`, `metrics`, `anti-mimicry`

---

## 3) Missing ground-truth LoRA/DreamBooth evaluation outputs

**Title**
`LoRA/DreamBooth benchmark: manifests/jobs exist, but no post-training anti-mimicry scoring`

**What is wrong with the benchmark**
LoRA and Anti-DreamBooth harnesses generate jobs and may execute train/infer commands, but there is no implemented evaluator that scores generated outputs against clean/protected baselines for anti-mimicry success.

**Why it can mislead users**
A successful job status (`completed`) can be mistaken for validated protection effectiveness. Execution success is not evaluation success.

**Evidence from the repo**
- LoRA harness job status tracks planned/completed/failed but no outcome metric computation: `src/auralock/benchmarks/lora.py:391-442`
- Anti-DreamBooth harness similarly tracks command execution status only: `src/auralock/benchmarks/antidreambooth.py:264-330`
- CLI marks execution failure only by preflight/job status, not by anti-mimicry result quality: `src/auralock/cli.py:769-777`, `src/auralock/cli.py:898-906`

**Proposed benchmark upgrade**
Implement a post-training evaluator stage:
- Generate fixed prompt/seed sample sets for clean vs protected training runs.
- Compute identity/style leakage metrics and prompt-consistency deltas.
- Emit a final “attack outcome report” with pass/fail thresholds.

**Acceptance criteria**
- [ ] Each executed job writes `evaluation.json` containing outcome metrics.
- [ ] Manifest includes an `evaluation` section (not only execution status).
- [ ] Tests validate that `--execute` produces both run artifacts and evaluation artifacts.

**Labels**
`benchmark`, `lora`, `dreambooth`, `evaluation`, `anti-mimicry`

---

## 4) Robustness suite misses common purification transforms

**Title**
`Robustness testing gap: no JPEG compression or crop transforms in protection-readability benchmark`

**What is wrong with the benchmark**
Robust transform suite currently includes identity, Gaussian blur, and resize-restore scales, but does not include JPEG compression artifacts or crop-based perturbation breakage tests.

**Why it can mislead users**
Protection can appear robust under current transforms while failing under common real-world pipelines (social media recompression, thumbnail crop, random crop augmentation).

**Evidence from the repo**
- Current transform suite: identity, blur, resize(0.75), resize(0.5): `src/auralock/core/style.py:217-231`
- No JPEG transform implementation in robustness suite; no crop transform in suite: `src/auralock/core/style.py:149-231`

**Proposed benchmark upgrade**
Extend transform suite and reporting with:
- JPEG compression at multiple quality levels (e.g., Q95/Q85/Q70).
- Center crop + random crop (then resize back).
- Optional stronger blur variants and resize kernels.

**Acceptance criteria**
- [ ] Transform suite includes JPEG and crop variants by name.
- [ ] Protection report contains per-transform metrics for all added transforms.
- [ ] Regression tests cover transform inclusion and metric key presence.

**Labels**
`benchmark`, `robustness`, `metrics`, `anti-mimicry`, `testing`

---

## 5) Dataset split integrity and leakage checks are missing

**Title**
`Dataset split risk: benchmark harness lacks leakage/overlap validation across split directories`

**What is wrong with the benchmark**
Subject split layout validates only folder existence and non-empty images. There is no check that set_A/set_B/set_C are disjoint by content/hash or that metadata leakage is prevented.

**Why it can mislead users**
Leakage between training/reference/holdout splits can inflate apparent protection or model behavior differences, invalidating benchmark conclusions.

**Evidence from the repo**
- Split validation only checks directory presence and image existence: `src/auralock/benchmarks/antidreambooth.py:110-149`
- Split handling copies files but does not deduplicate/verify disjointness: `src/auralock/benchmarks/antidreambooth.py:158-238`

**Proposed benchmark upgrade**
Add split-integrity validation before run:
- Hash-based duplicate detection across set_A/B/C.
- Optional near-duplicate detection (perceptual hash).
- Emit leak report and fail benchmark when leakage exceeds threshold.

**Acceptance criteria**
- [ ] Pre-run validation computes and stores split overlap stats.
- [ ] Execution is blocked when exact duplicates exist across forbidden split pairs.
- [ ] Tests cover duplicate-file and duplicate-content rejection.

**Labels**
`benchmark`, `data-quality`, `dataset`, `evaluation`, `integrity`

---

## 6) Reproducibility metadata is incomplete for benchmark claims

**Title**
`Reproducibility gap: benchmark outputs do not capture full run determinism metadata`

**What is wrong with the benchmark**
Reports/manifests omit critical reproducibility metadata (library versions, git commit, hardware/CUDA details, effective random seeds exposed at CLI, deterministic flags). Some defaults exist in config, but they are not fully surfaced in benchmark reports.

**Why it can mislead users**
Users may not be able to reproduce claimed numbers or compare results fairly across machines/runtimes.

**Evidence from the repo**
- `seed` exists in `LoraBenchmarkConfig` but is not exposed in CLI options for benchmark commands: `src/auralock/benchmarks/lora.py:71-74`, `src/auralock/cli.py:654-718`, `src/auralock/cli.py:780-841`
- Benchmark summary report focuses on aggregate metrics and omits environment provenance: `src/auralock/services/protection.py:183-192`

**Proposed benchmark upgrade**
Record and export reproducibility block in every report:
- Git SHA, command args, package versions, CUDA/torch info, seed(s), deterministic settings.
- Expose seed in CLI for all benchmark modes.

**Acceptance criteria**
- [ ] All benchmark reports include `reproducibility` object with commit/environment/seed details.
- [ ] CLI supports explicit `--seed` and writes resolved value into reports.
- [ ] Test verifies reproducibility object exists and is non-empty in generated reports.

**Labels**
`benchmark`, `reproducibility`, `infrastructure`, `research-quality`

---

## 7) Comparison methodology is weak (no baselines, no repeated trials, no stats)

**Title**
`Methodology gap: benchmark compares only internal profiles without external baselines or statistical confidence`

**What is wrong with the benchmark**
Current benchmark aggregates one-pass averages across AuraLock profiles (`safe/balanced/strong/...`) but does not include external baseline methods, repeated trials, confidence intervals, or significance testing.

**Why it can mislead users**
Single-run profile averages can reflect noise or implementation bias. Without baselines/statistics, claims of superiority or robustness are methodologically weak.

**Evidence from the repo**
- Profile benchmark computes per-profile means only: `src/auralock/services/protection.py:718-742`
- CLI benchmark accepts profile lists only (no baseline method matrix, no repeats/seeds): `src/auralock/cli.py:590-651`
- Existing benchmark tests validate profile summary existence, not comparative rigor/statistics: `src/tests/test_benchmark.py:21-44`

**Proposed benchmark upgrade**
Adopt comparative protocol:
- Include no-protection and at least one external reference defense baseline.
- Run N repeated trials per condition (fixed prompt/seed sets).
- Report mean ± std/CI and statistical tests for key outcomes.

**Acceptance criteria**
- [ ] Benchmark config supports repeated trials and baseline methods.
- [ ] Output report includes variance/CI fields and trial-level rows.
- [ ] Tests validate trial aggregation and baseline sections in output schema.

**Labels**
`benchmark`, `methodology`, `statistics`, `evaluation`, `anti-mimicry`
