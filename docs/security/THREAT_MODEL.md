# Threat model and security backlog (local-first)

**Scope:** This repository targets **local / clinical-research** use. Many controls required for internet-exposed SaaS or multi-tenant PHI are **not implemented**. This page consolidates the **`docs/todos/`** security and quality items into one place so reviewers do not have to walk ten separate files.

**Not a certification:** Completing items below does not equal HIPAA compliance; it is engineering hygiene and risk transparency.

## Assumptions (default)

- Operator runs the app on a **workstation or controlled network** they administer.
- Local FastAPI requires a per-launch bearer token and a loopback `Host` when started through `scripts/run_app.sh`; `make run-api` is unauthenticated dev mode. Streamlit has no authentication (see todo 003 and [LOCAL_DATA_SECURITY.md](LOCAL_DATA_SECURITY.md)).
- **CORS and bind addresses** are developer-friendly; harden before any network exposure.

## Risk register (summary)

Detailed write-ups remain in `docs/todos/` for history and acceptance criteria.

| ID | Priority | Topic | Detail file |
|----|----------|--------|-------------|
| 001 | P1 | SQL injection / unsafe SQL construction | [001-pending-p1-sql-injection-vulnerability.md](../todos/001-pending-p1-sql-injection-vulnerability.md) |
| 002 | P1 | Type hints / maintainability | [002-pending-p1-missing-type-hints-entire-codebase.md](../todos/002-pending-p1-missing-type-hints-entire-codebase.md) |
| 003 | P1 | No authentication or authorization | [003-pending-p1-no-authentication-authorization.md](../todos/003-pending-p1-no-authentication-authorization.md) |
| 004 | P1 | Outcome mapping validation | [004-pending-p1-outcome-mapping-validation-missing.md](../todos/004-pending-p1-outcome-mapping-validation-missing.md) |
| 005 | P1 | Path traversal (uploads / paths) | [005-pending-p1-path-traversal-vulnerability.md](../todos/005-pending-p1-path-traversal-vulnerability.md) |
| 006 | P1 | Statistical analysis coverage | [006-pending-p1-statistical-analysis-untested.md](../todos/006-pending-p1-statistical-analysis-untested.md) |
| 007 | P2 | Streamlit caching | [007-pending-p2-missing-streamlit-caching.md](../todos/007-pending-p2-missing-streamlit-caching.md) |
| 008 | P2 | Polars / pandas boundary overhead | [008-pending-p2-polars-pandas-conversion-overhead.md](../todos/008-pending-p2-polars-pandas-conversion-overhead.md) |
| 009 | P2 | PHI access / audit logging | [009-pending-p2-phi-access-audit-logging.md](../todos/009-pending-p2-phi-access-audit-logging.md) |
| 010 | P2 | Information disclosure in errors | [010-pending-p2-information-disclosure-errors.md](../todos/010-pending-p2-information-disclosure-errors.md) |

## Governed SQL (`POST /api/queries/sql`, off by default)

User-typed SQL runs through `src/clinical_analytics/sql/`. What it guarantees for **one answer**:

- **Read-only and sandboxed at the engine:** a fresh in-memory DuckDB per question whose only readable file is the version's stored Parquet (`allowed_paths`), with external access, extension loading and spilling off and the configuration locked. COPY, EXPORT, ATTACH, INSTALL/LOAD, `read_*`/`glob` and network reads fail even if the validator were bypassed (`tests/sql/test_sandbox.py`).
- **Checked against the contract:** one SELECT over `unified`, every node from an allowlist, every column a contract property; the patient ID is only counted. A PHI column is only filtered, counted, or grouped by its year, quarter or month, and only when the date column itself (or a CAST of it to a date or timestamp) is the direct argument of `year`, `quarter`, `month`, `EXTRACT` or `date_trunc` with that unit. Anything computed from it first (string building, date arithmetic, CASE, `datediff`) is refused, since it could carry the full value into a year (`tests/sql/test_privacy_adversarial.py`, PHI section). What runs is regenerated from the checked tree (comments dropped).
- **No count of 1-10 patients shown or derivable from this answer, or from it and any answer whose conditions it contains** (ADR013 section 4.5): the answer's conditions are every AND term of its WHERE, FILTERs and count_ifs, plus `IS NOT NULL` for each column summarised (at most 4; more are refused). Each is an indicator `coalesce(TRY(term), false)`, and one query counts patients per group and per combination of indicators (2^k cells per group, split per value of a coded column), over the whole table. `_suppress_lines` hides primary and secondary cells over the lines: a group's cells, each cell across groups, and the step-by-step count. A group's measures are shown only when none of its cells is hidden; an ungrouped answer with a hidden cell is withheld. That covers dropping any set of filters (not only one), differencing FILTERs or count_ifs inside one answer, and a filter that removes few patients in one group only (`tests/sql/test_privacy_adversarial.py`). Row-level expressions run inside `TRY` and measures in DOUBLE, so no data-dependent error can answer a yes/no question.

**Residual risks (not covered):**

- **Differences between answers whose conditions are not nested, or that group differently:** `count(A and B)` then `count(A and C)`, two GROUP BY expressions that differ by a few patients, or the same condition written two ways (`age > 65`, `65 < age`). An answer never reads earlier answers. The risk is per **dataset version**, not per session: anyone who can query the version can run the two questions in different chats, so the planned mitigation is a ledger of shown counts kept per dataset version (ADR013 section 4). A per-session ledger would be reset by starting a new chat.
- **The group-count limit** (`MAX_GROUPS`) refuses a query by how many groups it has, which a crafted grouping can make depend on a few patients; it is the same cross-query class as above.
- **Means of continuous columns** (more than 20 distinct values) reveal sums; many quantiles across queries trace values. Quantiles are limited to 0.1, 0.25, 0.5, 0.75 and 0.9.
- **Visit-level data** (an ID on more than one row) is refused, not supported.
- **Rows that IDs can't show to be patients** (no ID column, or rows without an ID) are refused until the user confirms one row per patient. The confirmation is the dataset's grain declaration (`contracts/declarations.py`, grain `patient`), kept per dataset version and row count. Suppression then rests on the user's word: if rows were in fact visits, a shown count of 11 or more rows could be fewer patients.

## Electron-specific notes

- Preload bridges the renderer to **local HTTP** (`VITE_API_URL`, default `http://127.0.0.1:8000`; never `localhost`, which may resolve to `::1`). Treat preload surface area as **security-sensitive** (expose only required IPC).
- Main process settings (for example `contextIsolation`, `sandbox`) are product decisions; document changes in an ADR when you change them.

## Tracking

Prefer **GitHub issues** (or your tracker) for assignable work, and keep this table in sync when an item closes or splits. The `docs/todos/` files remain the long-form audit trail until migrated.
