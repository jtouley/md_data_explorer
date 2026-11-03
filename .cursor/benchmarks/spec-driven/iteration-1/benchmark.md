# Command Benchmark: `/spec-driven`

**Date**: 2026-03-28
**Scope**: `.cursor/commands/spec-driven.md`
**Goal**: Ensure the command enforces meaningful tests and integration testing, not only TDD mechanics.

## Summary

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Meaningful test quality gate present | 0 | 1 | +1 |
| Integration-test enforcement gate present | 0 | 1 | +1 |
| Explicit anti-trivial-test prohibition | 0 | 1 | +1 |
| Commit evidence includes integration status | 0 | 1 | +1 |
| Aggregate rubric score (0-4) | 0 | 4 | +4 |

## Rubric

1. **Meaningful tests**: command requires behavior/invariant assertions and edge/failure coverage.
2. **Integration gate**: command mandates integration tests when changes cross boundaries.
3. **Anti-gaming**: command forbids trivial tests and mock-only substitution for integration.
4. **Evidence contract**: final quality-gate output must include integration-test result or explicit N/A rationale.

## Analyst notes

- Previous command focused on Red/Green/Refactor sequencing but allowed low-signal tests to satisfy the process.
- Updated command now blocks completion unless test quality and integration boundary coverage are addressed.
- This benchmark is static rubric-based (command-text quality benchmark), not runtime model-pass benchmarking.

## Artifacts

- `.cursor/commands/spec-driven.md`
- `.cursor/benchmarks/spec-driven/iteration-1/benchmark.json`
