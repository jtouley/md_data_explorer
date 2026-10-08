# Threat model and security backlog (local-first)

**Scope:** This repository targets **local / clinical-research** use. Many controls required for internet-exposed SaaS or multi-tenant PHI are **not implemented**. This page consolidates the **`docs/todos/`** security and quality items into one place so reviewers do not have to walk ten separate files.

**Not a certification:** Completing items below does not equal HIPAA compliance; it is engineering hygiene and risk transparency.

## Assumptions (default)

- Operator runs the app on a **workstation or controlled network** they administer.
- **No authentication** is enforced by default on Streamlit or local FastAPI (see todo 003).
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

## Electron-specific notes

- Preload bridges the renderer to **local HTTP** (`VITE_API_URL`, often `http://localhost:8000`). Treat preload surface area as **security-sensitive** (expose only required IPC).
- Main process settings (for example `contextIsolation`, `sandbox`) are product decisions; document changes in an ADR when you change them.

## Tracking

Prefer **GitHub issues** (or your tracker) for assignable work, and keep this table in sync when an item closes or splits. The `docs/todos/` files remain the long-form audit trail until migrated.
