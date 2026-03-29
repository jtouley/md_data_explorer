# Skill Benchmark: test-quality-loop

**Date**: 2026-03-23
**Evals**: eval-0 (quality loop plan), eval-1 (Playwright E2E plan), eval-2 (partial-success threshold) — one run per configuration each
**Outputs**: named folders under `iteration-1/` (`quality-loop-plan-monorepo/`, `playwright-e2e-streamlit/`, `resist-partial-success-threshold/`) plus `eval-*/**/grading.json` for aggregation.

## Summary

| Metric | With Skill | Without Skill | Delta |
|--------|------------|---------------|-------|
| Pass Rate | 100% ± 0% | 100% ± 0% | +0.00 |
| Time | 0.0s ± 0.0s | 0.0s ± 0.0s | +0.0s |
| Tokens | 0 ± 0 | 0 ± 0 | +0 |

## Analyst notes (discrimination)

- **Rubric outcome:** All listed expectations passed for **both** configurations on all three evals. The baseline model (no skill file) already produced staff-credible plans aligned with strong QA norms, so this benchmark **does not discriminate** skill vs. no-skill on pass rate.
- **Qualitative deltas (with-skill runs):** Tighter alignment to the skill’s **numbered phases**, explicit citation of **health-and-quality-report-template**, **merge-by-reproduction** rule, and **md_data_explorer**-specific paths (Streamlit iframe, `make test-e2e` vs browser) in the Playwright eval.
- **Next eval iterations:** Add scenarios where the baseline commonly fails: ambiguous scope (“fix all flakes”), pressure to skip `make test-*` in favor of raw pytest, or requests to add **many** shallow E2E tests without layering—then assert skill-specific remedies. Capture **token/time** from Task notifications on the next run for meaningful timing rows.
- **Trigger optimization:** Per skill-creator, if the goal is higher **invocation rate**, run `python -m scripts.run_loop` from `~/.cursor/skills/skill-creator` against a 20-query trigger set; pass rate here does not measure triggering.

## Artifacts

- `review.html` — static eval viewer (open locally in a browser).
- `evals/evals.json` — rubric source (under the skill directory): `~/.cursor/skills/test-quality-loop/evals/evals.json`.
