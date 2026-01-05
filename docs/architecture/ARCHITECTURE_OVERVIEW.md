# Architecture Overview: Semantic NL Query Platform

**Version:** 2.0
**Date:** 2025-12-24
**Status:** 🎯 Current Architecture + Strategic Direction

---

## 🏗️ System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Clinical Analytics Platform              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              User Interface Layer                   │     │
│  │                                                       │     │
│  │  • Free-form NL query input (Phase 3)              │     │
│  │  • Structured questions (existing)                  │     │
│  │  • Analysis results visualization                 │     │
│  │  • Data upload & management                        │     │
│  └──────────────────┬──────────────────────────────────┘     │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Question Engine (Enhanced)                   │     │
│  │                                                       │     │
│  │  ┌──────────────┐  ┌──────────────┐                │     │
│  │  │  NL Query    │  │  Structured  │                │     │
│  │  │  Parser      │  │  Questions   │                │     │
│  │  │              │  │  (Existing)  │                │     │
│  │  │  • Embedding │  │              │                │     │
│  │  │  • Intent    │  │  • Radio     │                │     │
│  │  │  • Entities  │  │    buttons  │                │     │
│  │  └──────────────┘  └──────────────┘                │     │
│  │           │                  │                       │     │
│  │           └────────┬─────────┘                       │     │
│  │                    ▼                                  │     │
│  │         ┌─────────────────────┐                      │     │
│  │         │  Intent Inference   │                      │     │
│  │         │  & Context Builder  │                      │     │
│  │         └─────────────────────┘                      │     │
│  └──────────────────┬──────────────────────────────────┘     │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Semantic Layer (Ibis)                       │     │
│  │                                                       │     │
│  │  • Config-driven outcomes, metrics, dimensions     │     │
│  │  • SQL generation from semantic understanding       │     │
│  │  • Query execution via DuckDB                      │     │
│  │  • Metadata for RAG context (Phase 3)              │     │
│  └──────────────────┬──────────────────────────────────┘     │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Dataset Registry                             │     │
│  │                                                       │     │
│  │  • Auto-discovery of datasets                       │     │
│  │  • Config-driven initialization                     │     │
│  │  • Zero-code dataset addition                       │     │
│  └──────────────────┬──────────────────────────────────┘     │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Data Layer                                  │     │
│  │                                                       │     │
│  │  • Persistent DuckDB storage (data/analytics.duckdb)│     │
│  │  • Parquet export for lazy Polars scanning          │     │
│  │  • Dataset versioning (content hash)                │     │
│  │  • CSV export (backward compatibility)              │     │
│  │  • Session recovery on app startup                   │     │
│  │  • Polars lazy frames for transformations           │     │
│  │  • DuckDB for query execution                       │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow: NL Query Processing

### Current Flow (Structured Questions)

```
User selects option → QuestionEngine.ask_initial_question()
    ↓
Radio button selection → Intent signal
    ↓
QuestionEngine.build_context_from_intent()
    ↓
AnalysisContext created → Variables selected
    ↓
SemanticLayer.get_cohort() → SQL generated
    ↓
Results displayed
```

### Enhanced Flow (NL Queries - Phase 3)

```
User types NL query → QuestionEngine.parse_natural_language_query()
    ↓
┌─────────────────────────────────────────┐
│  Semantic Understanding Layer           │
│                                         │
│  1. Embedding-based intent matching    │
│  2. Entity extraction from query        │
│  3. RAG context from semantic layer    │
│     (outcomes, variables, metrics)     │
└─────────────────┬───────────────────────┘
                  ↓
    AnalysisIntent + Entities extracted
                  ↓
    QuestionEngine.build_context_from_nl()
                  ↓
    AnalysisContext created (same as before)
                  ↓
    SemanticLayer.get_cohort() → SQL generated
                  ↓
    Results displayed
```

---

## 📦 Component Details

### 1. Question Engine

**Location:** `src/clinical_analytics/ui/components/question_engine.py`

**Current Capabilities:**
- Structured question flow (radio buttons)
- Intent inference from context
- Multi-turn conversation support
- AnalysisContext tracking

**Phase 3 Enhancements:**
- Free-form NL input parsing
- Embedding-based intent classification
- Entity extraction from NL queries
- RAG integration with semantic layer
- Hybrid NL + structured interface

**Key Classes:**
- `QuestionEngine` - Main orchestration
- `AnalysisContext` - State tracking
- `AnalysisIntent` - Intent enumeration

### 2. Semantic Layer

**Location:** `src/clinical_analytics/core/semantic.py`

**Current Capabilities:**
- Config-driven outcomes, metrics, dimensions
- SQL generation via Ibis
- Query execution via DuckDB
- Filter application
- Column mapping

**Phase 3 Role:**
- Provides metadata for RAG context
- Outcomes, variables, metrics available for entity matching
- Config structure enables zero-code NL understanding

**Key Methods:**
- `get_cohort()` - Generate and execute queries
- `get_base_view()` - Build Ibis expressions
- `apply_filters()` - Apply user filters

### 3. Dataset Registry

**Location:** `src/clinical_analytics/core/registry.py`

**Capabilities:**
- Auto-discovery of dataset implementations
- Config-driven initialization
- Factory pattern for dataset creation
- Zero-code dataset addition
- Unified upload handling (single-table = multi-table with 1 table)

**Integration:**
- Each dataset has semantic layer config
- Registry provides access to all semantic metadata
- Enables dataset-specific NL understanding
- All upload types use identical persistence and query capabilities

### 4. Storage Layer

**Location:** `src/clinical_analytics/storage/`

**Capabilities:**
- Persistent DuckDB storage at `data/analytics.duckdb` (ACID guarantees)
- Parquet export for lazy Polars scanning (columnar optimization)
- Dataset versioning (content hash for idempotent queries)
- CSV export (backward compatibility)
- Session recovery (restore datasets on app startup)
- Conversation history (JSONL audit trail)

**Boundary Rules:**
- **IO is eager**: File reads and DuckDB writes materialize data
- **Transforms are lazy**: All data transformations use Polars lazy frames
- **Semantics are declarative**: Semantic layer config is JSON/YAML

**Key Classes:**
- `DataStore` - Manages persistent DuckDB connection
- `QueryLogger` - JSONL conversation history
- `compute_dataset_version()` - Content hash for versioning

**Persistence Invariant:**
"Given the same upload hash + semantic config, results are immutable and reused."

### 5. Configuration System

**Location:** `data/configs/datasets.yaml`

**Structure:**
```yaml
dataset_name:
  # Metadata
  display_name: "..."
  status: "available"

  # Semantic layer config
  outcomes:
    outcome_name:
      source_column: "..."
      type: "binary"

  metrics:
    metric_name:
      expression: "..."
      label: "..."

  dimensions:
    dimension_name:
      label: "..."
      type: "categorical"

  # Column mappings
  column_mapping:
    source_col: target_col
```

**Phase 3 Usage:**
- Outcomes → Entity extraction targets
- Variables → Entity matching candidates
- Metrics → Query understanding context
- Dimensions → Grouping variable candidates

---

## 🔗 Integration Points

### NL Query → Semantic Layer

```python
# Phase 3: Enhanced QuestionEngine
class QuestionEngine:
    @staticmethod
    def parse_natural_language_query(
        query: str,
        semantic_layer: SemanticLayer
    ) -> AnalysisIntent:
        # 1. Get semantic layer metadata
        config = semantic_layer.config

        # 2. Extract available entities
        outcomes = list(config.get('outcomes', {}).keys())
        variables = list(config.get('column_mapping', {}).values())

        # 3. Use embeddings to match query to entities
        matched_outcome = match_entity(query, outcomes)
        matched_variables = match_entities(query, variables)

        # 4. Infer intent
        intent = infer_intent_from_query(query)

        return AnalysisIntent(
            intent=intent,
            outcome=matched_outcome,
            variables=matched_variables
        )
```

### Semantic Layer → SQL Generation

```python
# Existing: SemanticLayer generates SQL
cohort = semantic_layer.get_cohort(
    outcome_col="outcome_hospitalized",
    filters={"confirmed_only": True}
)

# Behind the scenes:
# 1. Reads config for outcome definition
# 2. Builds Ibis expression
# 3. Compiles to SQL
# 4. Executes via DuckDB
```

---

## 🎯 Design Principles

### 1. Config-Driven Everything
- No hardcoded logic
- All behavior from YAML config
- Zero-code dataset addition
- Semantic layer metadata drives NL understanding

### 2. Progressive Enhancement
- Structured questions work now
- NL queries enhance (don't replace)
- Hybrid approach for best UX
- Backward compatible

### 3. Semantic Understanding
- Use semantic layer as knowledge base
- RAG pattern for context
- Embedding-based matching
- Intent classification from NL

### 4. Separation of Concerns
- QuestionEngine: Query understanding
- SemanticLayer: SQL generation
- Registry: Dataset management
- UI: Presentation only

### 5. Boundary Rules

**IO is eager, transforms are lazy, semantics are declarative.**

- **IO Boundary**: File reads (`pl.read_csv()`, `pl.read_parquet()`) are eager. Use `pl.scan_csv()`/`pl.scan_parquet()` for lazy IO when possible. DuckDB writes materialize data.
- **Transform Boundary**: All data transformations use Polars lazy frames (`pl.LazyFrame`). Materialize only at query execution or UI render boundary.
- **Semantic Boundary**: Semantic layer config is declarative (JSON/YAML). No imperative logic in semantic layer initialization.
- **Documentation**: Any code that violates these boundaries must have explicit comment explaining why (e.g., Excel files require eager read due to Polars limitations).

**Persistence Invariant:**
"Given the same upload hash + semantic config, results are immutable and reused."

This means:
- Same content hash → same `dataset_version` → same DuckDB table → same query results
- Query execution uses `(upload_id, dataset_version)` as idempotent run key
- Semantic config changes create new version (different results)

---

## 📊 Current vs. Enhanced Architecture

| Aspect | Current | Enhanced (Phase 3) |
|--------|---------|-------------------|
| **Input Method** | Structured questions (radio buttons) | Free-form NL + structured fallback |
| **Intent Inference** | From context/answers | Embedding-based + context |
| **Entity Extraction** | Manual selection | Automatic from NL query |
| **Semantic Layer Usage** | SQL generation only | SQL generation + RAG context |
| **User Experience** | Guided but rigid | Natural language + guided |
| **Extensibility** | Config-driven | Config-driven + NL understanding |

---

## 🚀 Future Enhancements

1. **LLM-Based Classification** (Optional)
   - Claude/GPT for complex queries
   - Few-shot prompting
   - Fallback to embeddings

2. **Vector Store for RAG**
   - Embed semantic layer metadata
   - Similarity search
   - Query understanding cache

3. **Multi-Dataset Queries**
   - "Compare COVID-MS and Sepsis outcomes"
   - Cross-dataset entity matching

4. **Query Suggestions**
   - Auto-complete based on semantic layer
   - Query history learning

5. **Visual Query Builder**
   - Generate NL queries from visual selections
   - Bidirectional NL ↔ Visual

---

## 📝 Related Documentation

- **vision/UNIFIED_VISION.md** - Strategic direction and roadmap
- **architecture/IBIS_SEMANTIC_LAYER.md** - Semantic layer implementation details
- **research/NL_QUERY_BEST_PRACTICES.md** - Research foundation and patterns
- **architecture/ARCHITECTURE_REFACTOR.md** - Config-driven architecture evolution
- **implementation/IMPLEMENTATION_PLAN.md** - Phase-by-phase implementation plan

---

**This architecture enables natural language query capabilities while building on the existing config-driven, semantic layer foundation.**
