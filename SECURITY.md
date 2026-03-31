# Security Policy

## Supported versions

AuraLock is currently in **alpha** (`0.1.x`) and does not maintain long-term support branches.
Security fixes are typically applied to the latest commit on the default branch.

## Reporting a vulnerability

Please report suspected vulnerabilities privately.

- Do **not** create a public GitHub issue for sensitive security problems.
- Send a private report to the maintainers with:
  - affected component/file,
  - reproduction steps,
  - impact assessment,
  - and suggested mitigation (if available).

If you are unsure whether an issue is security-sensitive, report it privately first.

## What to include in a report

Provide as much detail as possible:

1. Vulnerability type (for example: path traversal, code execution, unsafe deserialization).
2. Exact version/commit tested.
3. Minimal proof-of-concept.
4. Expected vs actual behavior.
5. Practical impact and exploitation assumptions.

## Response targets (best effort)

- Initial acknowledgement: within **5 business days**.
- Triage and severity assessment: within **10 business days**.
- Fix timeline: depends on severity and reproducibility.

These targets are best-effort and may vary by maintainer availability.

## Disclosure process

- We will work with the reporter to validate and scope the issue.
- A fix will be prepared and reviewed.
- Public disclosure will happen after a patch is available, when feasible.

## Operational safety note

AuraLock is a research toolkit, not a production security boundary.
Do not rely on its outputs as a guarantee against all model extraction or imitation threats.
