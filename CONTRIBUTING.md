# Contributing Guide

Thanks for your interest in improving **AuraLock**.
This repository is research-oriented, so contributions are expected to prioritize:

- reproducibility,
- transparent trade-offs,
- and benchmark-backed claims.

## 1) Ways to contribute

- Fix bugs in protection, analysis, benchmark, or CLI flows.
- Improve tests and reliability for CPU-first workflows.
- Add documentation, examples, and experiment notes.
- Propose new profiles only when accompanied by measurable evidence.

## 2) Development setup

```bash
git clone https://github.com/VoDaiLocz/Lock-ART.
cd Lock-ART.

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e "[ui,dev]"
pip install -e "[benchmark,dev]"
```

## 3) Branch and commit conventions

- Create a focused branch for each change.
- Keep commits small and descriptive.
- Prefer commit messages in this structure:
  - `docs: ...`
  - `fix: ...`
  - `feat: ...`
  - `test: ...`
  - `refactor: ...`

Example:

```text
docs: add contributing and security documentation
```

## 4) Coding standards

- Python formatting: `black`.
- Linting: `ruff`.
- Type checks: `mypy` (best-effort with project config).
- Add/update tests for behavior changes.

Run before opening a PR:

```bash
ruff check src
black --check src
pytest
```

## 5) Research and benchmark integrity

When adding or changing protection logic:

- report both protection-oriented metrics and quality metrics,
- describe limitations clearly,
- avoid universal claims (for example, “blocks all models”),
- include enough context for another contributor to reproduce results.

Suggested minimum reporting fields:

- profile name and parameters,
- dataset/split description,
- PSNR, SSIM, and internal protection score,
- runtime environment notes (CPU/GPU, package versions).

## 6) Pull request checklist

Before requesting review, ensure:

- [ ] Scope is clear and limited.
- [ ] Documentation is updated (README/docs) when needed.
- [ ] Tests were added or updated for changed behavior.
- [ ] Local checks (`ruff`, `black --check`, `pytest`) were run.
- [ ] Any benchmark claims include reproducible context.

## 7) Documentation style

- Prefer concise, concrete instructions.
- Use command blocks for runnable steps.
- Explicitly label optional GPU-only workflows.
- Keep terminology consistent with existing profiles (`safe`, `balanced`, `strong`, `subject`, `fortress`, `blindfold`).

## 8) Security disclosures

Please do **not** open public issues for sensitive vulnerabilities.
Follow the private reporting process in [SECURITY.md](SECURITY.md).
