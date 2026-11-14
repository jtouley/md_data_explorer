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
- **Checked against the contract:** one SELECT over `unified`, every node from an allowlist, every column a contract property; the patient ID is only counted, PHI columns only coarsened or filtered. What runs is regenerated from the checked tree (comments dropped).
- **No count of 1-10 patients shown or derivable from this answer, or from it and the same query without its WHERE or without its GROUP BY:** the WHERE becomes a FILTER, so patients it removes are cells too; every aggregate's rows (FILTER, count_if, non-null values, each value of a column with at most 20 values) are cells; `_suppress_lines` hides primary and secondary cells. A measure is hidden when any of its cells is. Row-level expressions run inside `TRY` and measures in DOUBLE, so no data-dependent error can answer a yes/no question.

**Residual risks (not covered):**

- **Differences between different queries** beyond the two removals above: `count(A)` then `count(A and B)`, or two GROUP BY expressions that differ by a few patients. No single answer can detect them; a session ledger of shown counts is the planned mitigation.
- **The group-count limit** (`MAX_GROUPS`) refuses a query by how many groups it has, which a crafted grouping can make depend on a few patients; it is the same cross-query class as above.
- **Means of continuous columns** (more than 20 distinct values) reveal sums; many quantiles across queries trace values. Quantiles are limited to 0.1, 0.25, 0.5, 0.75 and 0.9.
- **Visit-level data** (an ID on more than one row) is refused, not supported.

## Electron-specific notes

- Preload bridges the renderer to **local HTTP** (`VITE_API_URL`, default `http://127.0.0.1:8000`; never `localhost`, which may resolve to `::1`). Treat preload surface area as **security-sensitive** (expose only required IPC).
- Main process settings (for example `contextIsolation`, `sandbox`) are product decisions; document changes in an ADR when you change them.

## Tracking

Prefer **GitHub issues** (or your tracker) for assignable work, and keep this table in sync when an item closes or splits. The `docs/todos/` files remain the long-form audit trail until migrated.
