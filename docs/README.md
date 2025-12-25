# Documentation Index

**Last Updated:** 2025-12-24  
**Status:** ✅ Consolidated around unified vision

---

## 🎯 Start Here

### [vision/UNIFIED_VISION.md](./vision/UNIFIED_VISION.md) ⭐ **READ THIS FIRST**
**The master document** consolidating all platform documentation around the strategic direction: enhancing QuestionEngine with semantic natural language query capabilities.

**Key Points:**
- Vision: Hybrid NL query + structured questions
- Architecture: Semantic layer + RAG patterns
- Roadmap: Phase-by-phase implementation
- Integration: How all components work together

---

## 📚 Core Documentation

### Vision & Strategy

1. **[vision/UNIFIED_VISION.md](./vision/UNIFIED_VISION.md)** ⭐ **READ THIS FIRST**
   - Strategic direction and roadmap
   - Integration philosophy
   - Phase-by-phase implementation

2. **[vision/SPECS_EVOLUTION_ANALYSIS.md](./vision/SPECS_EVOLUTION_ANALYSIS.md)**
   - Analysis: Evolution vs. departure from original specs
   - Comparison matrix: Original specs vs. unified vision
   - What specs can be deleted/archived
   - Branch strategy recommendations
   - **Verdict: 70% natural evolution, 30% UX departure**

### Architecture & Design

3. **[architecture/ARCHITECTURE_OVERVIEW.md](./architecture/ARCHITECTURE_OVERVIEW.md)**
   - System architecture diagrams
   - Component details and interactions
   - Data flow: NL query processing
   - Current vs. enhanced architecture

4. **[architecture/ARCHITECTURE_REFACTOR.md](./architecture/ARCHITECTURE_REFACTOR.md)**
   - Config-driven architecture evolution
   - From hardcoded to zero-code dataset addition
   - Registry pattern implementation
   - DRY principles applied

5. **[architecture/IBIS_SEMANTIC_LAYER.md](./architecture/IBIS_SEMANTIC_LAYER.md)**
   - Semantic layer implementation details
   - Config-driven outcomes, metrics, dimensions
   - SQL generation via Ibis
   - How to add new datasets

6. **[architecture/DATASET_STRUCTURE_PATTERNS.md](./architecture/DATASET_STRUCTURE_PATTERNS.md)**
   - Standard dataset file structure patterns across all datasets
   - COVID-MS, Sepsis, and MIMIC structure comparison
   - Data dictionary formats (README.txt, PDF, README.md)
   - How documentation structure enables NL query understanding
   - RAG context from data dictionaries
   - Entity extraction from documentation
   - Checklist for adding new datasets

### Implementation & Planning

7. **[implementation/IMPLEMENTATION_PLAN.md](./implementation/IMPLEMENTATION_PLAN.md)**
   - Phase-by-phase implementation plan
   - Security fixes (Phase 1)
   - Quality improvements (Phase 2)
   - **NL query enhancement (Phase 3)** ⭐
   - Additional improvements (Phase 4)

8. **[implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md](./implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md)** ⭐ **COMPREHENSIVE PLAN**
   - **17-day implementation plan** for question-driven semantic analysis
   - Phase 1: Documentation Infrastructure (MkDocs) - 2 days
   - Phase 2: Natural Language Query Engine - 5 days
   - Phase 3: Automatic Schema Inference - 3 days
   - Phase 4: Multi-Table Support - 4 days
   - Phase 5: Testing & Refinement - 3 days
   - Three-tier NL parsing: Pattern → Embeddings → LLM fallback
   - Auto-detect schemas, remove YAML configs
   - Complete transformation to question-driven interface

9. **[implementation/summary.md](./implementation/summary.md)**
   - Comprehensive code review summary
   - Current state assessment
   - Critical issues and priorities
   - Strategic direction reference

### Research & Best Practices

10. **[research/NL_QUERY_BEST_PRACTICES.md](./research/NL_QUERY_BEST_PRACTICES.md)** (2,795 lines)
    - Comprehensive research on NL query systems
    - Modern approaches: LLM-based, RAG, embeddings
    - Intent classification methods
    - Entity extraction techniques
    - Multi-turn conversation patterns
    - **Foundation for Phase 3 implementation**

### User Experience & Workflows

11. **[user-experience/PHASE_0_CLINICIAN_WORKFLOWS.md](./user-experience/PHASE_0_CLINICIAN_WORKFLOWS.md)**
    - Clinician-centered workflows
    - UI friction analysis
    - Self-service data upload
    - Publication-ready exports

12. **[user-experience/UI_FRICTION_ANALYSIS.md](./user-experience/UI_FRICTION_ANALYSIS.md)**
    - Detailed UX friction points
    - Usability improvements
    - Clinical researcher needs

13. **[user-experience/PLATFORM_CONSIDERATIONS.md](./user-experience/PLATFORM_CONSIDERATIONS.md)**
    - Platform-specific considerations
    - Deployment options
    - Infrastructure requirements

### Specifications

14. **[specs/spec_clinical_analytics_platform.md](./specs/spec_clinical_analytics_platform.md)**
    - Core platform specification
    - Multi-dataset support
    - Architecture overview
    - Execution plan

15. **[specs/IMPLEMENTATION_STATUS.md](./specs/IMPLEMENTATION_STATUS.md)**
    - Historical implementation status
    - Phase completion tracking
    - Architecture summary

16. **[specs/cursor-dry-refactor.md](./specs/cursor-dry-refactor.md)**
    - Semantic model refactor documentation
    - Evolution from hardcoded to config-driven

---

## 🔍 Quick Reference

### "I want to understand..."

**...the overall vision and direction**
→ Read [vision/UNIFIED_VISION.md](./vision/UNIFIED_VISION.md)

**...how the architecture works**
→ Read [architecture/ARCHITECTURE_OVERVIEW.md](./architecture/ARCHITECTURE_OVERVIEW.md)

**...how to implement NL queries**
→ Read [implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md](./implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md) (17-day plan)
→ Or [vision/UNIFIED_VISION.md](./vision/UNIFIED_VISION.md) Phase 1-3, then [research/NL_QUERY_BEST_PRACTICES.md](./research/NL_QUERY_BEST_PRACTICES.md)

**...the semantic layer**
→ Read [architecture/IBIS_SEMANTIC_LAYER.md](./architecture/IBIS_SEMANTIC_LAYER.md)

**...what needs to be fixed**
→ Read [implementation/summary.md](./implementation/summary.md) and [implementation/IMPLEMENTATION_PLAN.md](./implementation/IMPLEMENTATION_PLAN.md)

**...how to add a dataset**
→ Read [architecture/IBIS_SEMANTIC_LAYER.md](./architecture/IBIS_SEMANTIC_LAYER.md) "How to Add a New Dataset"

**...dataset file structure patterns**
→ Read [architecture/DATASET_STRUCTURE_PATTERNS.md](./architecture/DATASET_STRUCTURE_PATTERNS.md)
   - Understand COVID-MS, Sepsis, and MIMIC structure
   - See how documentation formats vary (txt, PDF, md)
   - Learn how data dictionaries enable NL queries

**...how data dictionaries enable NL queries**
→ Read [architecture/DATASET_STRUCTURE_PATTERNS.md](./architecture/DATASET_STRUCTURE_PATTERNS.md) "For NL Query Implementation"

**...NL query best practices**
→ Read [research/NL_QUERY_BEST_PRACTICES.md](./research/NL_QUERY_BEST_PRACTICES.md)

**...whether this is evolution or departure from original specs**
→ Read [vision/SPECS_EVOLUTION_ANALYSIS.md](./vision/SPECS_EVOLUTION_ANALYSIS.md)

---

## 📋 Document Status

| Document | Status | Last Updated | Purpose |
|----------|--------|--------------|---------|
| vision/UNIFIED_VISION.md | ✅ Current | 2025-12-24 | Strategic direction |
| vision/SPECS_EVOLUTION_ANALYSIS.md | ✅ Current | 2025-12-24 | Specs evolution analysis |
| architecture/ARCHITECTURE_OVERVIEW.md | ✅ Current | 2025-12-24 | System architecture |
| architecture/ARCHITECTURE_REFACTOR.md | ✅ Complete | 2025-12-07 | Architecture evolution |
| architecture/IBIS_SEMANTIC_LAYER.md | ✅ Complete | 2025-01-XX | Semantic layer docs |
| architecture/DATASET_STRUCTURE_PATTERNS.md | ✅ Current | 2025-12-24 | Dataset structure patterns |
| implementation/IMPLEMENTATION_PLAN.md | ✅ Updated | 2025-12-24 | Implementation roadmap |
| implementation/summary.md | ✅ Updated | 2025-12-24 | Code review summary |
| research/NL_QUERY_BEST_PRACTICES.md | ✅ Research | 2025-12-24 | Research foundation |
| user-experience/PHASE_0_CLINICIAN_WORKFLOWS.md | ✅ Current | 2025-12-24 | UX workflows |
| user-experience/UI_FRICTION_ANALYSIS.md | ✅ Current | 2025-12-24 | UX analysis |
| user-experience/PLATFORM_CONSIDERATIONS.md | ✅ Current | 2025-12-24 | Platform details |
| specs/spec_clinical_analytics_platform.md | ✅ Current | 2025-12-24 | Core specification |
| specs/IMPLEMENTATION_STATUS.md | ✅ Historical | 2025-12-07 | Implementation status |
| specs/cursor-dry-refactor.md | ✅ Historical | 2025-12-07 | Refactor documentation |

---

## 🎯 Key Concepts

### Semantic Natural Language Queries
Enhancing QuestionEngine to support free-form natural language input alongside structured questions, using semantic layer metadata for context-aware understanding.

### Semantic Layer
Config-driven layer that provides outcomes, metrics, dimensions, and relationships. Used for SQL generation and (in Phase 3) RAG context for NL queries.

### RAG Pattern
Retrieval-Augmented Generation: Using semantic layer metadata as knowledge base for NL query understanding, reducing errors by ~66% (Looker benchmark).

### Embedding-Based Intent Classification
Using sentence transformers (e.g., Sentence-BERT) to match user queries to canonical intents, achieving 85%+ accuracy.

### Hybrid Approach
Free-form NL queries as primary input, with structured questions as fallback for missing information or user preference.

---

## 🔗 Document Relationships

```
vision/UNIFIED_VISION.md (Master Document)
    │
    ├─→ architecture/ARCHITECTURE_OVERVIEW.md (How it works)
    │       └─→ architecture/IBIS_SEMANTIC_LAYER.md (Semantic layer details)
    │       └─→ architecture/ARCHITECTURE_REFACTOR.md (Evolution)
    │
    ├─→ implementation/IMPLEMENTATION_PLAN.md (How to build it)
    │       └─→ implementation/summary.md (What needs fixing)
    │
    ├─→ research/NL_QUERY_BEST_PRACTICES.md (Research foundation)
    │       └─→ Provides patterns for Phase 3
    │
    ├─→ architecture/DATASET_STRUCTURE_PATTERNS.md (Dataset patterns)
    │       └─→ How data dictionaries enable NL queries
    │       └─→ COVID-MS, Sepsis, MIMIC structure comparison
    │
    └─→ user-experience/PHASE_0_CLINICIAN_WORKFLOWS.md (User needs)
            └─→ user-experience/UI_FRICTION_ANALYSIS.md (UX details)
```

---

## 📝 Recent Changes (2025-12-24)

### Documentation Reorganization ✅

All documentation has been reorganized into logical directories:

- **vision/** - Strategic direction and evolution analysis
- **architecture/** - System design and technical architecture
- **implementation/** - Planning and execution guides
- **research/** - Research foundations and best practices
- **user-experience/** - UX analysis and workflows
- **specs/** - Technical specifications
- **todos/** - Action items and tasks
- **archive/** - Historical documentation

### Consolidation Complete ✅

All documentation has been consolidated around the unified vision:

1. **Created vision/UNIFIED_VISION.md** - Master document with strategic direction
2. **Created architecture/ARCHITECTURE_OVERVIEW.md** - System architecture with NL query integration
3. **Updated implementation/IMPLEMENTATION_PLAN.md** - Added Phase 3 (NL query enhancement)
4. **Updated implementation/summary.md** - Added strategic direction reference

### Key Alignment

All docs now align around:
> **Enhancing QuestionEngine with semantic natural language query capabilities that leverage our config-driven semantic layer using RAG patterns and embedding-based intent classification.**

### Dataset Structure Documentation

5. **Created architecture/DATASET_STRUCTURE_PATTERNS.md** - Documents consistent file structure patterns across COVID-MS, Sepsis, and MIMIC datasets, showing how data dictionaries (README.txt, PDF, README.md) enable NL query understanding and RAG context.

### Comprehensive Implementation Plan

6. **Created implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md** - 17-day comprehensive plan for transforming the platform to question-driven semantic analysis:
   - Documentation infrastructure with MkDocs
   - Three-tier NL query engine (Pattern → Embeddings → LLM)
   - Automatic schema inference (remove YAML configs)
   - Multi-table support for complex datasets
   - Complete testing and refinement

---

## 🚀 Next Steps

1. **Review vision/UNIFIED_VISION.md** - Understand the strategic direction
2. **Review architecture/ARCHITECTURE_OVERVIEW.md** - Understand how components integrate
3. **Review implementation/IMPLEMENTATION_PLAN.md Phase 3** - See NL query implementation plan
4. **Reference research/NL_QUERY_BEST_PRACTICES.md** - Use research patterns for implementation

---

**All documentation is now consolidated and aligned around the semantic NL query vision.** 🎯

