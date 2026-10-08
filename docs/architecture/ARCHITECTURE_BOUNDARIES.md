# Module boundaries (as-built)

Purpose: reduce guesswork when reviewing **imports and ownership** across `api/`, `core/`, `ui/`, and `datasets/`.

## Intended dependency direction

- **`clinical_analytics.core`**: Domain logic—semantic layer, NL query engine, overlays, query execution helpers, schemas. **No** FastAPI or Streamlit imports.
- **`clinical_analytics.datasets`**: Dataset definitions and factories (including uploaded datasets). May use `core` types.
- **`clinical_analytics.api`**: HTTP transport, Pydantic schemas, route handlers, small adapters. Should call into **`core`** and **`datasets`** for behavior.
- **`clinical_analytics.ui`**: Streamlit pages and components. May call **`core`** and **`datasets`** directly (today’s Streamlit path).

## Known couplings (review hotspots)

These are **real today**; tightening them is a migration concern tracked in the omnibus spec.

| From | To | Notes |
|------|-----|--------|
| `api/routes/datasets.py` | `ui.storage.user_datasets.UserDatasetStorage` | Upload path shares storage with Streamlit. |
| `api/routes/enrichments.py` | `ui.components.enrichment_integration.EnrichmentService` | Enrichment wiring crosses UI package. |

Prefer **new** API-only code to live under `api/services/` or `core/` rather than adding more `api` → `ui` imports unless the omnibus explicitly keeps a temporary bridge.

## Duplicate names

- **`QueryService`**: `clinical_analytics.core.query_service` (engine) vs `clinical_analytics.api.services.query_service` (async adapter). The API layer wraps core types (see `api/services/query_service.py`).

## How to verify

```bash
# Quick import scan (illustrative; extend patterns as needed)
rg '^from clinical_analytics|^import clinical_analytics' src/clinical_analytics/api -g'*.py'
```
