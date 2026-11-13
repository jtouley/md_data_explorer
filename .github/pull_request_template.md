## What and why

<!-- What does this change, and which issue does it close? (Closes #...) -->

## Privacy check

- [ ] No real patient data anywhere in this PR (code, tests, fixtures, screenshots, logs). Test data comes from the synthetic kit or the demo cohort.
- [ ] Nothing new can show a patient count from 1 to 10, or let one be worked out (differences, percentages, complementary filters, exports, logs, errors).
- [ ] Nothing new sends data to a non-loopback host.
- [ ] If any box above is unclear, the PR says why.

## Checks run

- [ ] `make format && make lint-fix`
- [ ] `make check-fast` (lint, format check, tracked-data guard, SPDX headers)
- [ ] `make type-check`
- [ ] `make test-fast` (or a narrower target, named here)
- [ ] Electron changes: `npm run lint` and `npm test` in `electron/`
- [ ] New behaviour has tests (AAA, `test_unit_scenario_expectedBehavior`, edge cases parametrized; see `tests/AGENTS.md`)

## Notes for the reviewer

<!-- Anything risky, anything left out, how you tested by hand. -->
