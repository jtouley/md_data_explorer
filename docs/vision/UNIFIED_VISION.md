# Unified Vision: Semantic Natural Language Query Platform

**Version:** 2.0
**Date:** 2025-12-24
**Status:** 🎯 Strategic Direction

---

## 🎯 Executive Summary

This document consolidates all platform documentation around a unified vision: **enhancing the Clinical Analytics Platform with semantic natural language query capabilities** that leverage our existing config-driven semantic layer architecture.

### Core Vision

Transform the platform from structured question-based analysis to a **hybrid system** that supports:
1. **Free-form natural language queries** with semantic understanding
2. **Structured questions** as a fallback/guided option
3. **Semantic layer integration** using RAG patterns for context-aware query understanding
4. **Zero-code dataset addition** (already achieved)
5. **Question-driven analysis** with intelligent intent inference

---

## 🏗️ Architecture Overview

### Current State (What We Have)

```
┌─────────────────────────────────────────────────────────────┐
│                    Current Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  Question    │      │  Semantic   │                     │
│  │  Engine      │─────▶│  Layer      │                     │
│  │              │      │  (Ibis)     │                     │
│  │  • Radio     │      │              │                     │
│  │    buttons   │      │  • Outcomes │                     │
│  │  • Structured│      │  • Metrics  │                     │
│  │    questions │      │  • Dimensions│                     │
│  └──────────────┘      └──────────────┘                     │
│         │                      │                             │
│         │                      │                             │
│         └──────────┬───────────┘                             │
│                    │                                           │
│              ┌─────▼─────┐                                   │
│              │  Dataset  │                                   │
│              │  Registry │                                   │
│              │  (Config) │                                   │
│              └───────────┘                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Strengths:**
- ✅ Config-driven semantic layer (Ibis-based)
- ✅ Zero-code dataset addition
- ✅ QuestionEngine with conversational flow
- ✅ Intent inference from context

**Gap:**
- ⚠️ Structured questions only (radio buttons)
- ⚠️ No free-form natural language input
- ⚠️ Semantic layer metadata not used for NL understanding

### Target State (Where We're Going)

```
┌─────────────────────────────────────────────────────────────┐
│              Enhanced Architecture with NL Queries           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Natural Language Query Interface            │     │
│  │                                                       │     │
│  │  ┌──────────────────┐      ┌──────────────────┐     │     │
│  │  │  Free-form NL    │      │  Structured     │     │     │
│  │  │  Input           │      │  Questions       │     │     │
│  │  │  (Primary)       │      │  (Fallback)     │     │     │
│  │  └──────────────────┘      └──────────────────┘     │     │
│  └──────────────┬──────────────────────┬─────────────────┘     │
│                 │                      │                         │
│                 ▼                      ▼                         │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Semantic Query Understanding Layer            │     │
│  │                                                       │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │     │
│  │  │  Embedding   │  │  Entity      │  │  Intent  │  │     │
│  │  │  Matching    │  │  Extraction  │  │  Classif.│  │     │
│  │  │              │  │              │  │          │  │     │
│  │  │  • Sentence  │  │  • Outcomes  │  │  • BERT  │  │     │
│  │  │    Transform │  │  • Variables │  │  • LLM   │  │     │
│  │  │  • Similarity│  │  • Groups    │  │          │  │     │
│  │  └──────────────┘  └──────────────┘  └──────────┘  │     │
│  └──────────────┬──────────────────────┬─────────────────┘     │
│                 │                      │                         │
│                 ▼                      ▼                         │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         RAG-Enhanced Context Retrieval                │     │
│  │                                                       │     │
│  │  • Semantic layer metadata (outcomes, variables)     │     │
│  │  • Variable synonyms and relationships               │     │
│  │  • Dataset-specific context                          │     │
│  └──────────────┬───────────────────────────────────────┘     │
│                 │                                               │
│                 ▼                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Enhanced Question Engine                      │     │
│  │                                                       │     │
│  │  • Intent inference from NL + semantic context       │     │
│  │  • Multi-turn conversation support                   │     │
│  │  • Missing information prompts                       │     │
│  └──────────────┬───────────────────────────────────────┘     │
│                 │                                               │
│                 ▼                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │         Semantic Layer (Ibis)                        │     │
│  │                                                       │     │
│  │  • Config-driven outcomes, metrics, dimensions       │     │
│  │  • SQL generation from semantic understanding       │     │
│  │  • Query execution                                  │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Integration Strategy

### Phase 1: Add Free-Form NL Input with Semantic Matching

**Goal:** Enable users to type natural language queries alongside structured questions.

**Implementation:**

```python
# Enhanced QuestionEngine with NL support
class QuestionEngine:
    @staticmethod
    def parse_natural_language_query(
        query: str,
        semantic_layer: SemanticLayer
    ) -> AnalysisIntent:
        """
        Parse free-form natural language using semantic understanding.

        Uses semantic layer metadata + embeddings for intent classification.
        """
        # 1. Semantic embeddings for intent matching
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-mpnet-base-v2')

        # Canonical intents (from research/NL_QUERY_BEST_PRACTICES.md)
        canonical_intents = {
            "compare_groups": "Compare outcomes between patient groups",
            "find_predictors": "Find variables that predict an outcome",
            "describe": "Describe patient characteristics and distributions",
            "survival": "Analyze time until an event occurs",
            "correlate": "Explore relationships between variables"
        }

        # Embed user query
        query_embedding = model.encode(query)

        # Embed canonical intents
        intent_embeddings = {
            k: model.encode(v) for k, v in canonical_intents.items()
        }

        # Find most similar intent
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = {
            intent: cosine_similarity(
                query_embedding.reshape(1, -1),
                emb.reshape(1, -1)
            )[0][0]
            for intent, emb in intent_embeddings.items()
        }

        best_intent = max(similarities, key=similarities.get)
        confidence = similarities[best_intent]

        # 2. Use semantic layer to extract entities
        entities = QuestionEngine.extract_entities(
            query,
            semantic_layer.config  # Your semantic layer config!
        )

        return AnalysisIntent(
            intent=best_intent,
            confidence=confidence,
            entities=entities
        )

    @staticmethod
    def extract_entities(
        query: str,
        semantic_config: dict
    ) -> dict:
        """
        Extract relevant variables from query using semantic layer metadata.

        This is where RAG pattern comes in - we use semantic layer
        as the knowledge base for entity extraction.
        """
        # Get available outcomes, variables, metrics from semantic layer
        available_outcomes = list(semantic_config.get('outcomes', {}).keys())
        available_variables = list(semantic_config.get('column_mapping', {}).values())
        available_metrics = list(semantic_config.get('metrics', {}).keys())

        # Use embeddings to match query terms to semantic layer entities
        # (Implementation details in research/NL_QUERY_BEST_PRACTICES.md)

        return {
            'outcome': matched_outcome,
            'predictors': matched_predictors,
            'grouping': matched_grouping
        }
```

**UI Integration:**

```python
# In Analyze.py page
def main():
    st.title("🔬 Analyze Your Data")

    # Option 1: Free-form natural language (primary)
    nl_query = st.text_input(
        "💬 Ask your research question:",
        placeholder="e.g., 'Do older patients have worse outcomes?'"
    )

    # Option 2: Structured questions (fallback)
    use_structured = st.checkbox("Or use guided questions instead")

    if nl_query and not use_structured:
        # Use semantic understanding
        intent = QuestionEngine.parse_natural_language_query(
            nl_query,
            semantic_layer=dataset.semantic  # Your semantic layer!
        )

        # Show what we understood
        st.info(f"✓ I understand: {intent.intent} (confidence: {intent.confidence:.0%})")

        # Prompt for missing info if needed
        if not intent.is_complete():
            missing = intent.get_missing_info()
            st.warning(f"I need to know: {', '.join(missing)}")
            # Show structured prompts for missing info

    elif use_structured:
        # Fall back to current structured approach
        intent_signal = QuestionEngine.ask_initial_question(cohort)
```

### Phase 2: RAG Integration with Semantic Layer

**Goal:** Use semantic layer metadata as context for better query understanding.

**Implementation:**

```python
class QuestionEngine:
    @staticmethod
    def parse_with_semantic_layer(
        query: str,
        semantic_layer: SemanticLayer
    ) -> AnalysisContext:
        """
        Use semantic layer metadata for RAG-based query understanding.

        This follows the Looker pattern from research/NL_QUERY_BEST_PRACTICES.md:
        - Semantic layer provides metadata (outcomes, variables, relationships)
        - RAG retrieves relevant context
        - LLM/embeddings generate structured intent
        """
        # 1. Get semantic layer metadata
        config = semantic_layer.config

        # 2. Build context from semantic layer
        semantic_context = {
            'outcomes': list(config.get('outcomes', {}).keys()),
            'variables': list(config.get('column_mapping', {}).values()),
            'metrics': list(config.get('metrics', {}).keys()),
            'dimensions': list(config.get('dimensions', {}).keys())
        }

        # 3. Use semantic matching to find relevant variables
        # (from research/NL_QUERY_BEST_PRACTICES.md embedding approach)
        relevant_variables = QuestionEngine.match_variables(
            query,
            semantic_context
        )

        # 4. Infer intent using semantic understanding
        intent = QuestionEngine.infer_intent_semantic(
            query,
            relevant_variables,
            semantic_context
        )

        # 5. Build AnalysisContext from semantic understanding
        context = AnalysisContext()
        context.inferred_intent = intent
        context.primary_variable = relevant_variables.get('outcome')
        context.predictor_variables = relevant_variables.get('predictors', [])
        # ... etc

        return context
```

### Phase 3: Hybrid Approach (Best UX)

**Goal:** Seamlessly combine free-form NL with structured questions.

**Features:**
- Free-form NL as primary input
- Structured questions for missing information
- Confidence-based prompting
- Multi-turn conversation support

---

## 📚 How This Aligns with Existing Documentation

### 1. research/NL_QUERY_BEST_PRACTICES.md (2,795 lines)
**Status:** ✅ Comprehensive research foundation

**Key Contributions:**
- Semantic embeddings approach (Sentence-BERT)
- RAG patterns with semantic layers (Looker example)
- Intent classification methods (BERT, LLM-based)
- Entity extraction techniques
- Multi-turn conversation patterns

**Integration:**
- Use embedding-based intent classification (Section 1)
- Apply RAG pattern with semantic layer as knowledge base (Section 1.2)
- Follow progressive disclosure patterns (Section 2)

### 2. IBIS_SEMANTIC_LAYER.md
**Status:** ✅ Implemented and ready

**Key Contributions:**
- Config-driven semantic layer architecture
- Outcomes, metrics, dimensions defined in YAML
- SQL generation via Ibis
- Zero-code dataset addition

**Integration:**
- Semantic layer provides metadata for NL query understanding
- Config entries become RAG context
- Outcomes/variables automatically available for entity extraction

### 3. ARCHITECTURE_REFACTOR.md
**Status:** ✅ Complete foundation

**Key Contributions:**
- Config-driven design eliminates hardcoding
- Registry pattern for dataset discovery
- DRY principles applied throughout

**Integration:**
- NL query enhancement builds on config-driven foundation
- No new hardcoding - everything uses semantic layer config
- Registry pattern enables dataset-specific NL understanding

### 4. QuestionEngine (Current Implementation)
**Status:** ✅ Conversational framework exists

**Key Contributions:**
- Structured question flow
- Intent inference from context
- Multi-turn conversation support
- AnalysisContext tracking

**Integration:**
- Enhance with NL input parsing
- Use existing intent inference logic
- Extend AnalysisContext with semantic entities
- Keep structured questions as fallback

### 5. IMPLEMENTATION_PLAN.md
**Status:** ⚠️ Needs update

**Current Focus:**
- Security fixes (SQL injection, auth)
- Type hints
- Performance optimization
- Testing

**Needs Addition:**
- Phase for NL query enhancement
- Integration with semantic layer
- Testing NL query accuracy

---

## 🎯 Implementation Roadmap

### Immediate (Week 1-2)

1. **Add Free-Form NL Input to QuestionEngine**
   - Text input field alongside structured questions
   - Basic embedding-based intent classification
   - Integration with existing AnalysisContext

2. **Semantic Layer Metadata Integration**
   - Extract outcomes, variables from semantic layer config
   - Use for entity matching in NL queries
   - Build RAG context from semantic layer

3. **Hybrid UI**
   - Primary: Free-form NL input
   - Fallback: Structured questions
   - Confidence-based prompting

### Short-term (Week 3-4)

4. **Enhanced Entity Extraction**
   - Variable matching using embeddings
   - Synonym support from semantic layer
   - Relationship inference

5. **Multi-Turn Conversation**
   - Missing information prompts
   - Query refinement
   - Context persistence

### Medium-term (Month 2-3)

6. **LLM-Based Classification (Optional)**
   - Claude/GPT integration for complex queries
   - Few-shot prompting with semantic layer context
   - Fallback to embeddings for simple queries

7. **Vector Store for RAG**
   - Embed semantic layer metadata
   - Enable similarity search
   - Cache query understanding results

---

## 🔗 MIMIC-IV Demo Integration Opportunity

The [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/hosp/#files-panel) provides:
- 100-patient demo dataset (open access)
- Hospital data (admissions, diagnoses, lab events)
- Perfect for testing NL query capabilities

### Dataset Structure (Similar to COVID-MS and Sepsis)

**MIMIC follows the same documentation pattern as COVID-MS and Sepsis:**

**COVID-MS includes:**
- `GDSI_OpenDataset_Final.csv` - Data file
- `README.txt` - Data dictionary and description
- `LICENSE.txt` - Usage license
- `SHA256SUMS.txt` - File integrity checksums

**Sepsis includes:**
- `training/*.psv` - Data files (PSV format, time-series)
- `physionet_challenge_2019_ccm_manuscript.pdf` - Data dictionary and methodology (PDF format)
- `LICENSE.txt` - ODbL license
- `SHA256SUMS.txt` - File integrity checksums

**MIMIC-IV Demo includes:**
- Data files (CSV/relational tables): `admissions.csv`, `patients.csv`, `diagnoses_icd.csv`, etc.
- **README files** - Comprehensive data dictionary with:
  - Table descriptions
  - Column definitions
  - Data types and formats
  - Relationships between tables
  - Usage guidelines
- Documentation files - Detailed schema documentation
- License files - Usage terms and conditions

**This consistent structure enables:**
- Automatic data dictionary parsing for semantic layer config (from README.txt, PDF, or README.md)
- Schema understanding from documentation files
- Variable name mapping from documentation
- Relationship inference from data dictionary
- RAG context for NL query understanding (Phase 3)

**Data Dictionaries:**
- Supplied by the user with their own upload (inside the ZIP, as a standalone PDF, or added to an existing upload)
- Extracted text is stored as `doc_context` in the upload's metadata in the per-user data home, stays local, and feeds NL query understanding and RAG context
- The repo ships no data dictionaries

### How to Leverage

```yaml
# data/configs/datasets.yaml
mimic4_demo:
  display_name: "MIMIC-IV Demo (100 patients)"
  source: "PhysioNet"
  status: "available"

  init_params:
    source_path: "data/raw/mimic4_demo/hosp/"

  # Semantic layer automatically understands:
  outcomes:
    mortality:
      source_column: "deathtime"  # From admissions table
      type: "binary"

  metrics:
    admission_count:
      expression: "COUNT(*)"
      label: "Number of Admissions"

    avg_los:
      expression: "AVG(los)"
      label: "Average Length of Stay"

  dimensions:
    diagnosis:
      label: "Primary Diagnosis"
      type: "categorical"
```

**Example NL Queries:**
- "What's the mortality rate by diagnosis?" → Semantic layer routes to appropriate query
- "Compare length of stay between ICU and non-ICU patients" → Intent: COMPARE
- "What predicts mortality in MIMIC-IV?" → Intent: FIND_PREDICTORS

---

## ✅ Success Criteria

### Phase 1 (Free-Form NL Input)
- [ ] Users can type natural language queries
- [ ] Intent classification accuracy ≥85% (from research/NL_QUERY_BEST_PRACTICES.md benchmarks)
- [ ] Semantic layer metadata used for entity extraction
- [ ] Structured questions remain as fallback

### Phase 2 (RAG Integration)
- [ ] Semantic layer config provides RAG context
- [ ] Variable matching accuracy ≥80%
- [ ] Query understanding errors reduced by 66% (Looker benchmark)

### Phase 3 (Hybrid UX)
- [ ] Seamless transition between NL and structured questions
- [ ] Multi-turn conversation support
- [ ] Confidence-based prompting for missing information

---

## 📊 Expected Benefits

### For Users (Clinical Researchers)
- ✅ Ask questions naturally instead of clicking through structured forms
- ✅ Faster analysis setup (seconds vs. minutes)
- ✅ More intuitive interface (matches how they think)
- ✅ Still have structured questions as fallback

### For Platform
- ✅ Leverages existing semantic layer investment
- ✅ Builds on config-driven architecture
- ✅ No new hardcoding - everything uses semantic layer config
- ✅ Extensible to new datasets automatically

### For Development
- ✅ Reuses existing QuestionEngine framework
- ✅ Semantic layer provides ready-made metadata
- ✅ NL_QUERY_BEST_PRACTICES provides proven approaches
- ✅ Incremental enhancement (not rewrite)

---

## 🔮 Future Enhancements

1. **Voice Input** - Speak queries instead of typing
2. **Query Suggestions** - Auto-complete based on semantic layer
3. **Query History** - Learn from previous queries
4. **Multi-Dataset Queries** - "Compare COVID-MS and Sepsis outcomes"
5. **Visual Query Builder** - Generate NL queries from visual selections
6. **Export NL Queries** - Save and share analysis workflows

---

## 📝 Document Alignment

This unified vision consolidates:

- ✅ **research/NL_QUERY_BEST_PRACTICES.md** → Provides research foundation and implementation patterns
- ✅ **architecture/IBIS_SEMANTIC_LAYER.md** → Provides metadata source for NL understanding
- ✅ **architecture/ARCHITECTURE_REFACTOR.md** → Provides config-driven foundation
- ✅ **QuestionEngine** → Provides conversational framework to enhance
- ✅ **implementation/IMPLEMENTATION_PLAN.md** → Will be updated with NL query phases
- ✅ **implementation/summary.md** → Will reflect this strategic direction

**All documentation now aligns around:**
> **Enhancing QuestionEngine with semantic natural language query capabilities that leverage our config-driven semantic layer using RAG patterns and embedding-based intent classification.**

---

## 🚀 Next Steps

1. **Review comprehensive plan** - See [implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md](../implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md) for detailed 17-day implementation plan
2. **Choose implementation approach**:
   - **Security-first**: [implementation/IMPLEMENTATION_PLAN.md](../implementation/IMPLEMENTATION_PLAN.md) - Fixes security issues first, then adds NL queries
   - **Feature-first**: [implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md](../implementation/plans/consolidate-docs-and-implement-question-driven-analysis.md) - 17-day plan focusing on question-driven transformation
3. **Update implementation/summary.md** - Reflect semantic NL query direction
4. **Prototype** - Build minimal NL input with embedding-based classification
5. **Test with MIMIC-IV Demo** - Validate approach with real dataset

---

**This unified vision brings together all our architectural investments (config-driven design, semantic layer, QuestionEngine) with modern NL query best practices to create a truly intuitive clinical analytics platform.**
