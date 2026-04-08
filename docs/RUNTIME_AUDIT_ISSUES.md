# Runtime Audit Issue Drafts

The following issue drafts focus only on runtime correctness, architectural risk, and user-visible trustworthiness.

---

## 1) Batch CLI returns success even when file-level failures occur

- **Severity:** High
- **Affected files:**  
  - `src/auralock/cli.py` (batch command)  
  - `src/auralock/services/protection.py` (`protect_directory`)
- **Exact risk:** `auralock batch` exits with code `0` even when `summary.failed_count > 0`, so automation cannot distinguish partial failure from full success.
- **User-visible impact:** CI jobs, scripts, and scheduled processing can report green status while some images were not protected.
- **Proposed change:** Return non-zero exit code when `failed_count > 0` (while still printing/saving failure details).
- **Acceptance criteria:**
  - A batch run with at least one failed file exits non-zero.
  - A fully successful batch run exits zero.
  - JSON report still includes detailed failures for failed runs.

---

## 2) Batch mode silently reports success for no-op runs

- **Severity:** Medium
- **Affected files:**  
  - `src/auralock/services/protection.py` (`_scan_directory_jobs`, `protect_directory`)  
  - `src/auralock/cli.py` (batch command)
- **Exact risk:** When all inputs are unsupported, skipped, or absent, batch processing can complete with `processed_count == 0` and exit success without a strong signal that no protection happened.
- **User-visible impact:** Users may believe images were protected even though output set is empty.
- **Proposed change:** Treat zero-processed runs as an explicit warning/error path (or require `--allow-empty` to keep exit code `0`).
- **Acceptance criteria:**
  - No-op batch runs produce a clear warning in console output.
  - Default behavior returns non-zero (or requires explicit opt-in for success on empty runs).
  - Report payload includes a machine-readable reason for zero processed files.

---

## 3) Adaptive protect reports do not expose attempted profile trail

- **Severity:** Medium
- **Affected files:**  
  - `src/auralock/services/protection.py` (`protect_file_adaptive`)  
  - `src/auralock/cli.py` (protect report serialization)
- **Exact risk:** Adaptive mode returns only the final selected result, with no per-profile attempt history in report output.
- **User-visible impact:** Users cannot verify whether escalation actually occurred, which profiles were tried, or why thresholds were missed/met.
- **Proposed change:** Include structured adaptive attempt metadata in report output (profile order, per-attempt metrics, constraint pass/fail, selected fallback reason).
- **Acceptance criteria:**
  - `--report` payload for adaptive mode contains an ordered attempt list.
  - Each attempt includes protection score, SSIM, PSNR, and pass/fail against thresholds.
  - Final selection reason is explicit when thresholds are not met.

---

## 4) Benchmark profile comparisons are not reproducible by default

- **Severity:** Medium
- **Affected files:**  
  - `src/auralock/services/protection.py` (`_build_attack`, benchmark methods)  
  - `src/auralock/attacks/pgd.py`  
  - `src/auralock/attacks/stylecloak.py`
- **Exact risk:** Stochastic attack paths (e.g., random starts) are not seeded/recorded in benchmark flow, so repeated benchmark runs can differ without traceability.
- **User-visible impact:** Reported profile wins/losses may drift between runs, reducing trust in benchmark conclusions.
- **Proposed change:** Add benchmark seed control and include seed + deterministic settings in report metadata.
- **Acceptance criteria:**
  - Benchmark CLI accepts a seed option.
  - Benchmark report includes seed/determinism metadata.
  - Re-running with same seed yields stable outputs for stochastic paths.

---

## 5) Benchmark execution aborts on first runtime error and drops partial evidence

- **Severity:** Medium
- **Affected files:**  
  - `src/auralock/services/protection.py` (`_collect_benchmark_entries`)  
  - `src/auralock/cli.py` (benchmark command behavior)
- **Exact risk:** Any single exception during benchmark collection interrupts the full run and discards remaining jobs instead of producing a partial-but-auditable report.
- **User-visible impact:** Long benchmark runs can fail late with no structured summary of completed work, making diagnosis and comparison difficult.
- **Proposed change:** Continue collecting per-image/profile entries after recoverable failures and return a summary with `failures` metadata plus non-zero exit code when failures exist.
- **Acceptance criteria:**
  - A benchmark run with one failing input still returns completed entries for successful inputs.
  - Report includes failure list with file/profile/error context.
  - CLI exits non-zero when any benchmark job fails.
