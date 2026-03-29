# Staff recommendation: “Skip Make; standardize on `uv run pytest tests/ui -q`”

## Bottom line

I would **not** adopt “skip Make entirely” as a team standard **unless** you replace it with **one documented, CI-identical command** (flags, markers, env vars, working directory) and you treat any drift from that command as a defect. `uv run pytest tests/ui -q` is a reasonable *local default* for a tight UI loop, but it is **not** a substitute for the repo’s quality gates unless it reproduces what merge and CI actually enforce.

## Why “just `pytest tests/ui -q`” is risky

- **Marker and config drift**: Project test runs are often filtered (`-m`, slow/integration exclusions, optional paths). Quiet mode hides skips and warnings that explain false confidence.
- **Environment drift**: `make test-*` targets typically encode the intended venv, `PYTHONPATH`, cwd, and sometimes extra args. `uv run` helps with the interpreter, but not necessarily parity with CI unless you mirror invocation.
- **False “we shipped faster”**: You can merge broken cross-module contracts, analysis/datasets regressions, and type/lint failures that UI tests never touch.
- **PR template omission**: Removing `make test-ui` without naming the **canonical** replacement invites every author to invent their own pytest incantation.

## What I would standardize instead (if the goal is speed *and* safety)

1. **Pick one canonical pre-merge command** that matches CI (or document CI as source of truth and add a script/`justfile`/thin `make check-fast` that only forwards to it—naming is secondary; **parity** is not).
2. **Keep a “fast inner loop”** explicitly labeled as non-merge-blocking, e.g. `uv run pytest tests/ui -q` **plus** whatever markers the team agrees on (often excluding `@pytest.mark.slow` / integration if that mirrors `test-fast`).
3. **Update PR template** to the **exact** command(s) authors must run before merge—not necessarily `make test-ui`, but **must** include the merge bar (see below).

## PR template wording (if Make is banned)

Replace “run `make test-ui`” with something like:

- **Required before merge**: paste the **same** command CI runs for PR checks (verbatim), e.g. “`uv run pytest <args matching CI>`” and “`uv run ruff check` / `uv run mypy`” if those are gates.
- **Optional fast loop**: “`uv run pytest tests/ui -q`” only if you document that it is **narrow** and **non-sufficient** alone.

Silence on `make test-ui` is fine; **silence on the merge bar** is not.

## What I would still run before merge (written plan)

Order is intentional: cheap failures first, then breadth.

1. **Format + lint** (as enforced on commit/CI): same tool versions as CI (`ruff` check/format or whatever the workflow uses).
2. **Type check** if CI blocks on it: `mypy` (or project equivalent) on the touched packages or full project per policy.
3. **Fast test slice** that matches the project’s definition of “pre-PR”: typically **all non-slow tests** or `test-fast` equivalent—not only `tests/ui`. UI-only runs miss core, datasets, analysis, and loader failures that still ship.
4. **Targeted tests** for the change: modules touched by the diff (not only directory-based guessing when behavior spans layers).
5. **Integration / slow** markers when the change touches persistence, DuckDB concurrency, file IO, or cross-service boundaries—either run them locally or accept that merge waits on CI but **do not** claim “green” on UI-only quiet runs.
6. **Coverage / regression gates** if the repo enforces them on push or in CI: run the same check locally or rely on CI but then **do not** merge on red.

## Explicit pushback on the release captain framing

“Ship faster” by narrowing the pre-merge bar **always** has a bill: more production defects, more revert churn, and slower *overall* delivery. If the team wants less Makefile friction, **delete friction by automating parity** (one script, documented args, CI calls it too)—not by shrinking the merge bar to `tests/ui -q`.

---

**Summary**: Allow `uv run pytest tests/ui -q` as a **local fast loop**; do **not** treat it as the team’s merge standard. Standardize on **one CI-identical command block** in the PR template and run **lint, types, fast full-suite (or equivalent), and targeted integration** before merge.
