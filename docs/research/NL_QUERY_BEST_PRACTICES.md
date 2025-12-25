# Natural Language Query Best Practices for Clinical Data Analysis

**Research Date:** 2025-12-24
**Domain Focus:** Clinical research, healthcare analytics, conversational data interfaces

---

## Executive Summary

This document synthesizes best practices for building natural language query interfaces for research and analysis, specifically focused on clinical data analysis tools. The research draws from modern BI tools (Tableau, Looker, ThoughtSpot, Mode), NLU frameworks, and academic research on intent classification and conversational AI.

### Key Findings

1. **Modern NLQ implementations favor LLM-based semantic understanding** over rule-based pattern matching for production systems
2. **Progressive disclosure and multi-turn conversations** improve user experience for complex analytical queries
3. **Semantic layers (like Looker's LookML)** reduce AI errors by up to 66% when combined with RAG architectures
4. **Intent classification using transformer models (BERT, RoBERTa)** achieves 85%+ accuracy for analytical query categorization
5. **Session management and state tracking** are critical for multi-step conversations and query refinement

---

## 1. Natural Language Question Parsing

### Architecture Patterns

#### **Translation Layer Pattern** (ThoughtSpot Spotter)
Modern NLQ systems use a multi-layered architecture:
- **Input Layer**: Raw natural language text
- **Translation Layer**: Converts natural language to database syntax
- **RAG Layer**: Retrieval-Augmented Generation for context
- **Reasoning Layer**: BARQ (Business Analytics Reasoning & Query) for domain logic
- **Trust Layer**: Security, validation, and confidence scoring

```
User Query → NLP Service → Context Enrichment → Query Generation → Validation → Results
```

#### **Semantic Layer Pattern** (Looker Conversational Analytics)
Uses existing semantic models to constrain query generation:
- LLM translates natural language to **Explore queries** (not raw SQL)
- Semantic layer provides:
  - Data model metadata (tables, joins, relationships)
  - Business logic and calculations
  - Access controls and security rules
- **Result**: 66% reduction in errors compared to direct SQL generation

```python
# Example: RAG-based semantic layer query
# From Twilio's Looker + Amazon Bedrock implementation

# 1. User query
query = "What were ICU admissions last month by diagnosis?"

# 2. Retrieve relevant LookML metadata from vector store
semantic_context = vector_store.similarity_search(
    query,
    filter={"type": "looker_view", "domain": "clinical"}
)

# 3. Generate Looker Explore query (not SQL)
explore_query = llm.generate(
    prompt=f"""Convert to Looker query:
    Query: {query}
    Available fields: {semantic_context}
    Output: Looker Explore parameters"""
)

# 4. Execute through Looker API
results = looker_api.run_explore(explore_query)
```

### Query Understanding Components

#### **Token-Based Pattern Matching** (Tableau Ask Data)
- Keyword-based system maps intent to analytical query
- Breaks utterances into tokens using context
- Dynamic suggestions update as phrases are added
- **Synonym configuration** for field names (e.g., "revenue" → "sales", "income", "earnings")

#### **Embedding-Based Semantic Matching**
The meaning of a sentence is captured by averaging word embeddings:
- "I want to compare mortality rates" ≈ "Show me death rate differences"
- Pre-trained embeddings (BERT, Sentence-BERT) enable zero-shot understanding
- Small training datasets (50-100 examples) sufficient for domain adaptation

```python
# Example: Semantic similarity for query understanding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# Define canonical query patterns
canonical_queries = {
    "compare_groups": "Compare outcomes between patient groups",
    "predict_outcome": "Predict patient outcomes based on variables",
    "describe_distribution": "Show distribution of values",
    "correlate_variables": "Analyze relationship between variables"
}

# Embed user query
user_query = "What's the difference in recovery rates between treatment groups?"
query_embedding = model.encode(user_query)

# Find most similar intent
canonical_embeddings = {k: model.encode(v) for k, v in canonical_queries.items()}
similarities = {k: cosine_similarity(query_embedding, v)
                for k, v in canonical_embeddings.items()}

intent = max(similarities, key=similarities.get)
# Result: "compare_groups" (similarity: 0.87)
```

### Data Preparation for NLQ

**Must-Have Practices** (from Tableau Ask Data whitepaper):
1. **Simplify data sources** - Minimal set of fields users will query
2. **Remove/hide unnecessary fields** - Reduces ambiguity
3. **Data curation by stewards** - Users more successful with curated data
4. **Field naming conventions** - Use natural, business-friendly names
5. **Synonym definitions** - Map domain terminology to field names

**Example Synonym Configuration:**
```yaml
# Clinical data synonyms
admissions:
  synonyms: [admits, hospitalizations, stays, encounters]

diagnosis:
  synonyms: [condition, disease, illness, icd, diagnostic_code]

mortality:
  synonyms: [death, deceased, died, fatal, fatality_rate]

los:
  synonyms: [length_of_stay, hospital_days, admission_duration, stay_length]
```

---

## 2. Intent Classification Patterns

### Four Main Approaches (2025 Landscape)

#### **1. Rule-Based Systems** (Legacy)
- **Pros**: Fast, interpretable, no training data needed
- **Cons**: Brittle, doesn't generalize, high maintenance
- **Use Case**: Prototypes, tightly controlled domains, demos
- **Example**: Regex patterns for exact keyword matching

```python
# Simple rule-based intent classifier
import re

def classify_intent_rules(query):
    if re.search(r'\b(compare|difference|versus|vs)\b', query, re.I):
        return "COMPARE"
    elif re.search(r'\b(predict|forecast|will|likelihood)\b', query, re.I):
        return "PREDICT"
    elif re.search(r'\b(correlat|relationship|association)\b', query, re.I):
        return "CORRELATE"
    elif re.search(r'\b(describe|distribution|show|what are)\b', query, re.I):
        return "DESCRIBE"
    else:
        return "UNKNOWN"
```

#### **2. Classical Machine Learning**
- **Models**: SVM, Random Forest, Logistic Regression, Naive Bayes
- **Features**: TF-IDF, n-grams, POS tags
- **Pros**: Interpretable, works with small datasets
- **Cons**: Manual feature engineering, limited contextual understanding
- **Use Case**: Mid-sized datasets (500-5000 examples), interpretability required

```python
# Classical ML intent classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Training data
X_train = [
    "Compare mortality between treatment groups",
    "What's the difference in recovery rates?",
    "Predict readmission risk for diabetic patients",
    "Correlation between BMI and complications"
]
y_train = ["COMPARE", "COMPARE", "PREDICT", "CORRELATE"]

# Pipeline
classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
    ('clf', LinearSVC(C=1.0, class_weight='balanced'))
])

classifier.fit(X_train, y_train)

# Prediction
query = "Show relationship between age and outcome"
intent = classifier.predict([query])[0]  # "CORRELATE"
```

#### **3. Transformer-Based Models** (Production Standard)
- **Models**: BERT, RoBERTa, DistilBERT, XLM-RoBERTa (multilingual)
- **Performance**: 85%+ accuracy on intent classification
- **Pros**: State-of-the-art accuracy, handles context, transfer learning
- **Cons**: Requires GPU, more complex deployment
- **Use Case**: Production systems, high accuracy requirements

```python
# BERT fine-tuning for intent classification
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# Define intents for clinical analytics
intents = {
    0: "COMPARE",      # Compare groups/cohorts
    1: "PREDICT",      # Predictive modeling
    2: "DESCRIBE",     # Descriptive statistics
    3: "CORRELATE",    # Association/correlation analysis
    4: "AGGREGATE",    # Counting, summing, averaging
    5: "FILTER",       # Subset/filter data
    6: "TREND"         # Time-series analysis
}

# Prepare training data
train_data = [
    {"text": "Compare survival rates between treatment arms", "label": 0},
    {"text": "What factors predict ICU admission?", "label": 1},
    {"text": "Show me the distribution of ages in cohort", "label": 2},
    {"text": "Is there a relationship between smoking and complications?", "label": 3},
    {"text": "How many patients were readmitted?", "label": 4},
    {"text": "Show only diabetic patients over 65", "label": 5},
    {"text": "What is the trend in mortality over time?", "label": 6},
]

dataset = Dataset.from_list(train_data)

# Load pre-trained model
model_name = "bert-base-uncased"  # or "emilyalsentzer/Bio_ClinicalBERT" for clinical
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(intents)
)

# Tokenize
def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir='./intent_classifier',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Inference
def classify_intent(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    confidence = torch.softmax(logits, dim=-1).max().item()

    return {
        "intent": intents[predicted_class],
        "confidence": confidence
    }

# Example
result = classify_intent("Is age associated with longer hospital stays?")
# {'intent': 'CORRELATE', 'confidence': 0.94}
```

#### **4. LLM-Based Classification** (Emerging)
- **Models**: GPT-4, Claude, Gemini with few-shot prompting
- **Performance**: 90%+ accuracy with good prompting
- **Pros**: No training, flexible, can extract entities simultaneously
- **Cons**: Latency, cost, requires API access
- **Use Case**: Rapid prototyping, complex domain reasoning

```python
# LLM-based intent classification with structured output
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an intent classifier for clinical research queries.

Classify queries into these intents:
- COMPARE: Compare groups, treatments, or cohorts
- PREDICT: Predictive modeling or forecasting
- DESCRIBE: Descriptive statistics, distributions
- CORRELATE: Association or correlation analysis
- AGGREGATE: Counting, summing, averaging
- FILTER: Subset or filter data
- TREND: Time-series analysis

Also extract key entities:
- variables: Clinical variables mentioned
- cohorts: Patient groups or subsets
- time_period: Temporal references
- statistical_test: If a specific test is implied

Return JSON with intent, confidence (0-1), and entities."""

def classify_query_llm(query: str):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Classify this query:\n\n{query}"}
        ]
    )

    # Parse JSON response
    import json
    return json.loads(response.content[0].text)

# Example
query = "Do patients over 65 with diabetes have longer ICU stays compared to younger patients?"
result = classify_query_llm(query)

# Result:
# {
#   "intent": "COMPARE",
#   "confidence": 0.95,
#   "entities": {
#     "variables": ["icu_length_of_stay"],
#     "cohorts": ["patients_over_65_with_diabetes", "younger_patients"],
#     "time_period": null,
#     "statistical_test": "t-test or Mann-Whitney U test"
#   }
# }
```

### Multi-Turn Intent Classification

Modern systems track intent across multiple exchanges:

```python
# Session-aware intent classifier
class ConversationalIntentClassifier:
    def __init__(self):
        self.conversation_history = []
        self.current_intent = None
        self.context = {}

    def classify(self, query: str):
        # Include conversation history for context
        context_queries = [msg['query'] for msg in self.conversation_history[-3:]]
        full_context = "\n".join(context_queries + [query])

        # Classify with context
        result = classify_intent(full_context)

        # Track conversation
        self.conversation_history.append({
            'query': query,
            'intent': result['intent'],
            'confidence': result['confidence']
        })

        return result

# Example multi-turn conversation
classifier = ConversationalIntentClassifier()

# Turn 1
classifier.classify("Show me diabetic patients")
# Intent: FILTER, confidence: 0.92

# Turn 2 - "their" refers to diabetic patients from Turn 1
classifier.classify("What is their average age?")
# Intent: AGGREGATE, confidence: 0.88, context resolved

# Turn 3 - "them" refers to diabetic patients
classifier.classify("Compare them to non-diabetic patients")
# Intent: COMPARE, confidence: 0.94, context resolved
```

### Data Augmentation for Intent Classification

When training data is limited:

```python
# Techniques from 2025 research
augmentation_techniques = {
    "backtranslation": "Translate to another language and back",
    "paraphrasing": "Use LLM to generate paraphrases",
    "synonym_replacement": "Replace entities/verbs with synonyms",
    "llm_generation": "Generate synthetic examples with LLMs"
}

# Example: LLM-based data augmentation
def augment_training_data(example_query, intent, num_variations=5):
    prompt = f"""Generate {num_variations} variations of this clinical research query
    that have the same intent but different wording:

    Original: "{example_query}"
    Intent: {intent}

    Variations should:
    - Use different medical terminology
    - Vary sentence structure
    - Maintain the same analytical intent
    """

    # Use LLM to generate variations
    variations = llm.generate(prompt)
    return variations

# Example
original = "Compare mortality rates between treatment groups"
variations = augment_training_data(original, "COMPARE")

# Generated variations:
# - "What's the difference in death rates across treatment arms?"
# - "How do survival outcomes differ between the intervention groups?"
# - "Is there a significant variance in fatality between cohorts?"
# - "Analyze mortality disparities across experimental conditions"
```

---

## 3. Pattern Matching vs Semantic Understanding

### Decision Framework

| Criteria | Pattern Matching | Semantic Embeddings | Transformer Models | LLM-Based |
|----------|------------------|---------------------|--------------------|-----------|
| **Training Data** | None | 50-200 examples | 500-5000 examples | None (few-shot) |
| **Accuracy** | 60-70% | 75-85% | 85-95% | 90-98% |
| **Handles Variations** | Poor | Good | Excellent | Excellent |
| **Setup Time** | Hours | Days | Weeks | Hours |
| **Inference Speed** | <1ms | 5-10ms | 50-100ms | 500-2000ms |
| **Cost** | Minimal | Low | Medium | High |
| **Interpretability** | High | Medium | Low | Medium |

### When to Use Each Approach

#### **Pattern Matching**
✅ Use when:
- Building MVP/prototype
- Domain has limited, fixed vocabulary (<50 query types)
- Need millisecond response times
- No training data available
- Complete transparency required

❌ Avoid when:
- Users express queries in many different ways
- Domain terminology varies
- Need to handle synonyms and paraphrases

#### **Semantic Embeddings**
✅ Use when:
- Starting production conversational AI project
- Have 50-200 labeled examples
- Need robustness to linguistic variation
- Moderate latency acceptable (10-50ms)
- Want balance of performance and cost

❌ Avoid when:
- Need state-of-the-art accuracy
- Have very large training dataset (use transformers instead)

#### **Transformer Models (BERT, etc.)**
✅ Use when:
- Production system with high accuracy requirements
- Have 500+ labeled examples (or can generate synthetic data)
- Can deploy GPU infrastructure
- Latency budget 50-200ms
- **This is the modern go-to for production NLP**

❌ Avoid when:
- Have <100 training examples
- Need <10ms response time
- Limited computational resources

#### **LLM-Based Classification**
✅ Use when:
- Rapid prototyping (no training needed)
- Complex reasoning required (e.g., multi-hop intent)
- Want to extract entities + classify intent simultaneously
- Can tolerate 500ms-2s latency
- Cost is not primary concern

❌ Avoid when:
- Need consistent sub-100ms response times
- Processing millions of queries/day
- Data privacy prevents external API calls

### Hybrid Approach (Recommended for Production)

```python
class HybridIntentClassifier:
    """
    Combine multiple approaches for optimal accuracy and latency
    """
    def __init__(self):
        # Fast rule-based filter for high-confidence patterns
        self.rule_classifier = RuleBasedClassifier()

        # BERT model for most queries
        self.bert_classifier = BertIntentClassifier()

        # LLM fallback for ambiguous cases
        self.llm_classifier = LLMIntentClassifier()

    def classify(self, query: str):
        # Step 1: Try rules (1ms)
        rule_result = self.rule_classifier.classify(query)
        if rule_result['confidence'] > 0.95:
            return rule_result

        # Step 2: Use BERT (50ms)
        bert_result = self.bert_classifier.classify(query)
        if bert_result['confidence'] > 0.85:
            return bert_result

        # Step 3: Fall back to LLM for ambiguous queries (1000ms)
        llm_result = self.llm_classifier.classify(query)
        return llm_result

# Performance characteristics:
# - 80% of queries handled by rules/BERT (<100ms)
# - 20% of queries use LLM fallback (~1s)
# - Overall average: ~250ms
# - Accuracy: 95%+
```

---

## 4. Conversational UI Flows

### Progressive Disclosure Pattern

Progressive disclosure reveals information gradually, keeping essential content in primary UI and advanced features in secondary UI.

#### **Application to Data Analysis**

```
┌─────────────────────────────────────┐
│ 1. Initial Query (Simple)           │
├─────────────────────────────────────┤
│ User: "Show diabetic patients"      │
│                                      │
│ System: [Shows patient list]        │
│   Found 247 patients with diabetes  │
│                                      │
│   💡 What would you like to know?   │
│   • Average age and demographics    │
│   • Medication patterns             │
│   • Comorbidities                   │
│   • Lab values                      │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 2. Progressive Refinement           │
├─────────────────────────────────────┤
│ User: "Average age and demographics"│
│                                      │
│ System: [Shows summary statistics]  │
│   Average age: 62.4 years           │
│   Gender: 54% F, 46% M              │
│   Race: ...                          │
│                                      │
│   ↓ Show advanced statistics        │
│     (std dev, confidence intervals) │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 3. Deep Dive (On Demand)            │
├─────────────────────────────────────┤
│ User: [Clicks "Show advanced"]      │
│                                      │
│ System:                              │
│   Mean: 62.4 (SD: 11.2)             │
│   95% CI: [61.0, 63.8]              │
│   Median: 64, IQR: [54, 72]         │
│   Distribution: [Shows histogram]   │
│                                      │
│   🔍 Test for normality?            │
│   📊 Compare to another group?      │
└─────────────────────────────────────┘
```

### Common UI Components

From 2025 research on progressive disclosure:

1. **"Show more" links** - Simplest form, reveals additional information
2. **Accordions** - User controls when to expand content (ideal for FAQs, options)
3. **Tabs** - Organize content into categories (reduces scrolling)
4. **Tooltips/Popovers** - On-hover information without leaving task
5. **Multi-step forms** - Break complex workflows into stages

#### **Example: Multi-Step Query Builder**

```python
# Progressive disclosure in query building
class ProgressiveQueryBuilder:
    def __init__(self):
        self.query_state = {
            "intent": None,
            "cohort": None,
            "variables": [],
            "stratification": None,
            "time_period": None
        }

    def process_step(self, user_input, step):
        if step == 1:  # Clarify intent
            return {
                "prompt": "What would you like to do?",
                "options": [
                    "Compare groups",
                    "Predict outcomes",
                    "Explore relationships",
                    "Summarize data"
                ],
                "next_step": 2
            }

        elif step == 2:  # Define cohort
            self.query_state["intent"] = user_input
            return {
                "prompt": "Which patients should I include?",
                "suggestions": [
                    "All patients",
                    "Patients with [condition]",
                    "Patients in [time range]",
                    "Custom filter..."
                ],
                "next_step": 3
            }

        elif step == 3:  # Select variables
            self.query_state["cohort"] = user_input
            return {
                "prompt": "What variables are you interested in?",
                "suggestions": self.get_relevant_variables(
                    self.query_state["intent"]
                ),
                "advanced_options": "Show all variables (250+)",
                "next_step": 4
            }

        # ... continue building query progressively

# Example interaction
builder = ProgressiveQueryBuilder()

# Step 1
response = builder.process_step(None, step=1)
# Shows: "Compare groups | Predict outcomes | ..."

# Step 2
response = builder.process_step("Compare groups", step=2)
# Shows: "Which patients? All | With diabetes | ..."

# Step 3
response = builder.process_step("Patients with diabetes", step=3)
# Shows: "Variables? HbA1c | Medications | Complications | ..." (relevant to diabetes)
```

### Dialog Management Patterns

#### **State Machine Approach**

Each conversation state represents an intent with nested state machines for subtasks.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class ConversationState(Enum):
    INITIAL = "initial"
    INTENT_CLASSIFICATION = "intent_classification"
    COHORT_DEFINITION = "cohort_definition"
    VARIABLE_SELECTION = "variable_selection"
    ANALYSIS_CONFIRMATION = "analysis_confirmation"
    RESULTS_PRESENTATION = "results_presentation"
    REFINEMENT = "refinement"

@dataclass
class DialogContext:
    state: ConversationState
    intent: Optional[str]
    entities: dict
    conversation_history: List[dict]
    current_query: Optional[dict]

class DialogManager:
    def __init__(self):
        self.context = DialogContext(
            state=ConversationState.INITIAL,
            intent=None,
            entities={},
            conversation_history=[],
            current_query=None
        )

    def handle_message(self, user_message: str):
        """Main dialog flow controller"""

        if self.context.state == ConversationState.INITIAL:
            return self.handle_initial_query(user_message)

        elif self.context.state == ConversationState.INTENT_CLASSIFICATION:
            return self.handle_intent_clarification(user_message)

        elif self.context.state == ConversationState.COHORT_DEFINITION:
            return self.handle_cohort_refinement(user_message)

        # ... handle other states

    def handle_initial_query(self, message: str):
        # Classify intent
        intent_result = classify_intent(message)

        if intent_result['confidence'] > 0.9:
            # High confidence - proceed
            self.context.intent = intent_result['intent']
            self.context.state = ConversationState.COHORT_DEFINITION
            return self.prompt_for_cohort()
        else:
            # Low confidence - clarify
            self.context.state = ConversationState.INTENT_CLASSIFICATION
            return self.prompt_for_intent_clarification(intent_result)

    def prompt_for_intent_clarification(self, intent_result):
        top_intents = intent_result['top_k']  # Top 3 intents

        return {
            "message": "I'm not quite sure what you're looking for. Did you want to:",
            "options": [
                {"label": intent['label'], "value": intent['intent']}
                for intent in top_intents
            ],
            "allow_free_text": True
        }

# Example usage
dm = DialogManager()

# Initial query (ambiguous)
response = dm.handle_message("Look at diabetes patients and outcomes")
# System asks: "Did you want to: [Compare] [Predict] [Describe]?"

# User clarifies
response = dm.handle_message("Compare")
# System moves to: "Which outcomes? [Mortality] [Readmission] [Complications]?"
```

### Multi-Turn Conversation Patterns

#### **Context Retention Pattern**

```python
class ConversationContext:
    """Maintains context across turns"""

    def __init__(self):
        self.entities = {}  # Extracted entities across turns
        self.active_cohort = None
        self.active_variables = []
        self.last_result = None

    def resolve_reference(self, query: str):
        """Resolve pronouns and references to previous context"""

        # Example: "What about their age?" -> "their" = active_cohort
        if "their" in query or "them" in query:
            if self.active_cohort:
                query = query.replace("their", f"the {self.active_cohort}'s")
                query = query.replace("them", f"the {self.active_cohort}")

        # Example: "Compare it to control" -> "it" = last result variable
        if "it" in query and self.active_variables:
            query = query.replace("it", self.active_variables[-1])

        return query

# Example
context = ConversationContext()

# Turn 1
context.active_cohort = "diabetic patients"
query1 = "Show diabetic patients"

# Turn 2 - pronoun reference
query2 = "What is their average age?"
resolved2 = context.resolve_reference(query2)
# "What is the diabetic patients's average age?"

# Turn 3 - implicit comparison
context.active_variables.append("average age")
query3 = "Compare it to non-diabetic patients"
resolved3 = context.resolve_reference(query3)
# "Compare average age to non-diabetic patients"
```

#### **Follow-Up Question Pattern**

```python
class FollowUpGenerator:
    """Suggests relevant follow-up questions based on results"""

    def generate_followups(self, intent: str, results: dict):
        if intent == "DESCRIBE":
            return [
                f"Compare {results['variable']} across groups?",
                f"Show distribution by demographics?",
                f"Test for statistical significance?"
            ]

        elif intent == "COMPARE":
            return [
                "Adjust for confounding variables?",
                "Perform subgroup analysis?",
                "Show confidence intervals?",
                "Export detailed results?"
            ]

        elif intent == "CORRELATE":
            return [
                "Visualize relationship?",
                "Control for covariates?",
                "Test for non-linear relationship?",
                "Stratify by subgroups?"
            ]

# Example
fug = FollowUpGenerator()

# After showing comparison results
results = {"variable": "mortality", "p_value": 0.023}
followups = fug.generate_followups("COMPARE", results)

# Display to user:
# "✓ Comparison complete (p=0.023)"
# "What next?"
# • Adjust for confounding variables?
# • Perform subgroup analysis?
# • Show confidence intervals?
```

---

## 5. Mapping Natural Language to Statistical Tests

### Research Findings

A 2025 study evaluated LLMs (ChatGPT, Google Bard, Microsoft Bing Chat, Perplexity) on selecting appropriate statistical tests:

**Performance Results:**
- ChatGPT 3.5: 85.19% concordance, 100% acceptance
- Microsoft Bing Chat: 96.3% concordance, 100% acceptance
- Google Bard: 77.78% concordance, 96.3% acceptance
- Perplexity: 85.19% concordance, 100% acceptance

**Key Insight:** All LLMs showed >75% concordance with expert statisticians, suggesting LLM-based test selection is viable for production systems.

### Intent → Statistical Test Mapping

```python
# Clinical analytics intent-to-test mapping
STATISTICAL_TEST_MAP = {
    "COMPARE": {
        "two_groups": {
            "continuous_normal": "independent_t_test",
            "continuous_non_normal": "mann_whitney_u",
            "categorical": "chi_square",
            "paired": "paired_t_test"
        },
        "multiple_groups": {
            "continuous_normal": "anova",
            "continuous_non_normal": "kruskal_wallis",
            "categorical": "chi_square"
        }
    },

    "CORRELATE": {
        "both_continuous": {
            "linear": "pearson_correlation",
            "monotonic": "spearman_correlation"
        },
        "one_categorical": "point_biserial",
        "both_categorical": "chi_square"
    },

    "PREDICT": {
        "continuous_outcome": "linear_regression",
        "binary_outcome": "logistic_regression",
        "time_to_event": "cox_regression",
        "count_outcome": "poisson_regression"
    },

    "TREND": {
        "single_group": "repeated_measures_anova",
        "time_series": "arima",
        "survival": "kaplan_meier"
    }
}

def select_statistical_test(
    intent: str,
    variables: dict,
    data_characteristics: dict
) -> dict:
    """
    Select appropriate statistical test based on intent and data

    Args:
        intent: Classified query intent (COMPARE, CORRELATE, etc.)
        variables: Dict with 'outcome', 'predictors', 'grouping'
        data_characteristics: Dict with 'normality', 'sample_size', etc.

    Returns:
        Dict with recommended test, assumptions, and alternatives
    """

    if intent == "COMPARE":
        n_groups = len(variables.get('groups', []))
        outcome_type = variables['outcome']['type']
        is_paired = data_characteristics.get('paired', False)

        if n_groups == 2:
            if outcome_type == "continuous":
                if is_paired:
                    return {
                        "test": "paired_t_test",
                        "assumptions": ["normality of differences"],
                        "alternative": "wilcoxon_signed_rank"
                    }
                elif data_characteristics.get('normality', True):
                    return {
                        "test": "independent_t_test",
                        "assumptions": ["normality", "equal variance"],
                        "alternative": "mann_whitney_u"
                    }
                else:
                    return {
                        "test": "mann_whitney_u",
                        "assumptions": ["none (non-parametric)"],
                        "alternative": "permutation_test"
                    }

            elif outcome_type == "categorical":
                return {
                    "test": "chi_square" if data_characteristics['sample_size'] > 40
                            else "fisher_exact",
                    "assumptions": ["expected frequency ≥5 per cell"],
                    "alternative": "fisher_exact"
                }

        elif n_groups > 2:
            if outcome_type == "continuous":
                if data_characteristics.get('normality', True):
                    return {
                        "test": "one_way_anova",
                        "assumptions": ["normality", "homogeneity of variance"],
                        "post_hoc": "tukey_hsd",
                        "alternative": "kruskal_wallis"
                    }

    elif intent == "CORRELATE":
        var1_type = variables['var1']['type']
        var2_type = variables['var2']['type']

        if var1_type == "continuous" and var2_type == "continuous":
            if data_characteristics.get('linear_relationship', True):
                return {
                    "test": "pearson_correlation",
                    "assumptions": ["linearity", "bivariate normality"],
                    "alternative": "spearman_correlation"
                }
            else:
                return {
                    "test": "spearman_correlation",
                    "assumptions": ["monotonic relationship"],
                    "interpretation": "measures monotonic association"
                }

    elif intent == "PREDICT":
        outcome_type = variables['outcome']['type']
        n_predictors = len(variables.get('predictors', []))

        if outcome_type == "continuous":
            return {
                "test": "multiple_linear_regression" if n_predictors > 1
                        else "simple_linear_regression",
                "assumptions": [
                    "linearity",
                    "independence of residuals",
                    "homoscedasticity",
                    "normality of residuals"
                ],
                "diagnostics": ["residual plots", "vif for multicollinearity"]
            }

        elif outcome_type == "binary":
            return {
                "test": "logistic_regression",
                "assumptions": ["independence of observations"],
                "metrics": ["auc_roc", "accuracy", "sensitivity", "specificity"]
            }

        elif outcome_type == "time_to_event":
            return {
                "test": "cox_proportional_hazards",
                "assumptions": ["proportional hazards"],
                "diagnostics": ["schoenfeld residuals"],
                "alternative": "parametric survival models"
            }

    return {"test": "unknown", "recommendation": "consult statistician"}

# Example usage
query_analysis = {
    "intent": "COMPARE",
    "variables": {
        "outcome": {"name": "mortality", "type": "categorical"},
        "groups": ["treatment_a", "treatment_b"]
    },
    "data_characteristics": {
        "sample_size": 100,
        "paired": False
    }
}

test_selection = select_statistical_test(
    query_analysis['intent'],
    query_analysis['variables'],
    query_analysis['data_characteristics']
)

# Result:
# {
#   "test": "chi_square",
#   "assumptions": ["expected frequency ≥5 per cell"],
#   "alternative": "fisher_exact"
# }
```

### LLM-Based Test Selection (Advanced)

For complex scenarios, use LLM reasoning:

```python
def select_test_with_llm(
    query: str,
    intent: str,
    variables: dict,
    data_summary: dict
) -> dict:
    """
    Use LLM to reason about appropriate statistical test
    """

    prompt = f"""You are a biostatistician helping select an appropriate statistical test.

Query: "{query}"
Intent: {intent}

Variables:
- Outcome: {variables['outcome']['name']} ({variables['outcome']['type']})
- Groups/Predictors: {variables.get('groups') or variables.get('predictors')}

Data Characteristics:
- Sample size: {data_summary['n']}
- Distribution: {data_summary.get('normality', 'unknown')}
- Missing data: {data_summary.get('missing_pct', 0)}%

Select the most appropriate statistical test and provide:
1. Primary test recommendation
2. Assumptions that need to be checked
3. Alternative test if assumptions violated
4. Code snippet to run the test

Return as JSON."""

    response = llm.generate(prompt)
    return json.loads(response)

# Example
result = select_test_with_llm(
    query="Do diabetic patients have higher readmission rates than non-diabetic patients?",
    intent="COMPARE",
    variables={
        "outcome": {"name": "readmitted_30d", "type": "binary"},
        "groups": ["diabetic", "non_diabetic"]
    },
    data_summary={
        "n": 500,
        "normality": "n/a",
        "missing_pct": 2.1
    }
)

# LLM Response:
# {
#   "primary_test": "chi_square_test",
#   "rationale": "Comparing proportions of a binary outcome between two groups",
#   "assumptions": [
#     "Independent observations",
#     "Expected frequency ≥5 in all cells"
#   ],
#   "alternative": "fisher_exact_test (if small cell counts)",
#   "code": "from scipy.stats import chi2_contingency\n..."
# }
```

### Decision Tree for Test Selection

```
Is it a COMPARISON?
├─ Yes
│  ├─ How many groups?
│  │  ├─ 2 groups
│  │  │  ├─ Continuous outcome?
│  │  │  │  ├─ Normal distribution? → Independent t-test
│  │  │  │  └─ Non-normal? → Mann-Whitney U
│  │  │  └─ Categorical outcome? → Chi-square / Fisher's exact
│  │  └─ 3+ groups
│  │     ├─ Continuous outcome?
│  │     │  ├─ Normal? → One-way ANOVA
│  │     │  └─ Non-normal? → Kruskal-Wallis
│  │     └─ Categorical outcome? → Chi-square
│
├─ Is it a CORRELATION?
│  ├─ Both continuous?
│  │  ├─ Linear relationship? → Pearson correlation
│  │  └─ Non-linear/monotonic? → Spearman correlation
│  └─ One categorical, one continuous? → Point-biserial
│
├─ Is it a PREDICTION?
│  ├─ Continuous outcome? → Linear regression
│  ├─ Binary outcome? → Logistic regression
│  ├─ Time-to-event? → Cox regression / Kaplan-Meier
│  └─ Count outcome? → Poisson regression
│
└─ Is it a TREND over time?
   ├─ Single group repeated measures? → Repeated measures ANOVA
   ├─ Time series? → ARIMA / Time series regression
   └─ Survival over time? → Kaplan-Meier / Log-rank test
```

---

## 6. Progressive Disclosure in Question-Driven Interfaces

### Design Principles

From 2025 UX research:

1. **Defer Advanced Features** - Keep essential content in primary UI, advanced options in secondary UI
2. **User Control** - Users decide when to reveal more information
3. **Reduce Cognitive Load** - Don't overwhelm with all options upfront
4. **Task-Oriented** - Progressive disclosure should follow natural task flow

### Implementation Patterns

#### **1. Accordion Pattern for Query Options**

```jsx
// React component example
function QueryBuilder({ onQuerySubmit }) {
  const [expanded, setExpanded] = useState({
    basicFilters: true,
    advancedFilters: false,
    statisticalOptions: false
  });

  return (
    <div className="query-builder">
      {/* Always visible: Basic query */}
      <section>
        <h3>What would you like to analyze?</h3>
        <textarea
          placeholder="Ask a question in plain English..."
          rows={3}
        />
      </section>

      {/* Progressive disclosure: Basic filters */}
      <Accordion
        title="Filters"
        expanded={expanded.basicFilters}
        defaultExpanded={true}
      >
        <DateRangeFilter />
        <PatientGroupFilter />
        <DemographicFilters />
      </Accordion>

      {/* Progressive disclosure: Advanced */}
      <Accordion
        title="Advanced Options"
        expanded={expanded.advancedFilters}
        badge="Optional"
      >
        <MultiVariateFilters />
        <CustomCalculations />
        <DataQualityOptions />
      </Accordion>

      {/* Progressive disclosure: Statistical */}
      <Accordion
        title="Statistical Settings"
        expanded={expanded.statisticalOptions}
        badge="Expert"
      >
        <SignificanceLevel />
        <MultipleTestingCorrection />
        <MissingDataHandling />
      </Accordion>

      <button onClick={onQuerySubmit}>
        Run Analysis
      </button>
    </div>
  );
}
```

#### **2. Tooltip Pattern for Contextual Help**

```jsx
function StatisticalTestResult({ result }) {
  return (
    <div className="result-card">
      <h4>
        Chi-square Test
        <Tooltip content="Tests independence between categorical variables">
          <InfoIcon />
        </Tooltip>
      </h4>

      <div className="metric">
        <span>χ² = {result.statistic}</span>
        <Tooltip content="Chi-square test statistic. Higher values indicate stronger association.">
          <InfoIcon />
        </Tooltip>
      </div>

      <div className="metric">
        <span>p-value = {result.pvalue}</span>
        <Tooltip content="Probability of observing this result if there's no association. p < 0.05 typically considered significant.">
          <InfoIcon />
        </Tooltip>
      </div>

      {/* Progressive disclosure: Show details on demand */}
      <ExpandableSection title="Show calculation details">
        <pre>{result.calculation_details}</pre>
      </ExpandableSection>

      <ExpandableSection title="Assumptions & diagnostics">
        <AssumptionChecks checks={result.assumptions} />
      </ExpandableSection>
    </div>
  );
}
```

#### **3. Multi-Step Wizard Pattern**

```python
class AnalysisWizard:
    """
    Multi-step wizard for complex analyses with progressive disclosure
    """

    STEPS = [
        "define_question",      # Step 1: What do you want to know?
        "select_cohort",        # Step 2: Which patients?
        "choose_variables",     # Step 3: What variables?
        "configure_analysis",   # Step 4: How to analyze?
        "review_and_run"        # Step 5: Review & confirm
    ]

    def __init__(self):
        self.current_step = 0
        self.wizard_state = {}

    def get_current_step_ui(self):
        step = self.STEPS[self.current_step]

        if step == "define_question":
            return {
                "title": "What would you like to investigate?",
                "description": "Describe your research question in plain language",
                "input": {
                    "type": "textarea",
                    "placeholder": "e.g., Do diabetic patients have longer hospital stays?"
                },
                "hints": [
                    "Compare two groups",
                    "Predict an outcome",
                    "Find relationships",
                    "Describe patterns"
                ],
                "buttons": ["Next →"]
            }

        elif step == "select_cohort":
            # Parse question to suggest relevant cohorts
            suggested_cohorts = self.extract_cohorts_from_question(
                self.wizard_state['question']
            )

            return {
                "title": "Define your patient cohort",
                "description": "Who should be included in the analysis?",
                "suggestions": suggested_cohorts,
                "input": {
                    "type": "cohort_builder",
                    "options": {
                        "prebuilt": ["All patients", "Recent admissions", "..."],
                        "custom": "Build custom filter..."
                    }
                },
                "preview": f"{self.estimate_cohort_size()} patients match criteria",
                "buttons": ["← Back", "Next →"]
            }

        elif step == "choose_variables":
            return {
                "title": "Select variables to analyze",
                "description": "What factors do you want to examine?",
                "smart_suggestions": self.suggest_variables(
                    self.wizard_state['question'],
                    self.wizard_state['cohort']
                ),
                "input": {
                    "type": "multi_select",
                    "grouped": True,  # Group by category
                    "searchable": True
                },
                "progressive_disclosure": {
                    "show_all_variables": "See all 250+ variables",
                    "advanced_transformations": "Create calculated fields"
                },
                "buttons": ["← Back", "Next →"]
            }

        elif step == "configure_analysis":
            # Automatically suggest test based on previous inputs
            suggested_test = self.suggest_statistical_test(
                self.wizard_state['question'],
                self.wizard_state['variables']
            )

            return {
                "title": "Configure analysis settings",
                "description": f"Recommended: {suggested_test['name']}",
                "auto_configured": True,
                "sections": [
                    {
                        "title": "Statistical Test",
                        "default": suggested_test,
                        "allow_override": True,
                        "expanded": False  # Collapsed by default
                    },
                    {
                        "title": "Significance Level",
                        "default": 0.05,
                        "expanded": False
                    },
                    {
                        "title": "Missing Data Handling",
                        "default": "listwise_deletion",
                        "expanded": False,
                        "badge": "Advanced"
                    }
                ],
                "info": "Default settings work for most analyses. Click to customize.",
                "buttons": ["← Back", "Review →"]
            }

        elif step == "review_and_run":
            return {
                "title": "Review your analysis",
                "summary": self.build_analysis_summary(),
                "editable": True,  # Click any section to edit
                "estimated_time": "< 1 minute",
                "buttons": ["← Back", "Run Analysis"]
            }

    def suggest_variables(self, question: str, cohort: dict):
        """Use LLM to suggest relevant variables"""
        # Extract mentioned concepts from question
        # Return prioritized list of relevant variables
        pass

# Example usage
wizard = AnalysisWizard()

# Step 1
ui = wizard.get_current_step_ui()
# Shows: Simple question input with hints

wizard.submit_step({"question": "Do diabetic patients have longer stays?"})

# Step 2 - Auto-suggests: "Patients with diabetes" and "All patients"
ui = wizard.get_current_step_ui()

wizard.submit_step({"cohort": "patients_with_diabetes"})

# Step 3 - Auto-suggests relevant variables: length_of_stay, admission_date, etc.
ui = wizard.get_current_step_ui()

# ... continue through wizard
```

#### **4. Smart Defaults with "Show More" Pattern**

```python
class SmartDefaultsUI:
    """
    Provide intelligent defaults with option to override
    """

    def render_analysis_config(self, analysis_spec: dict):
        """
        Show analysis configuration with smart defaults
        """

        # Auto-configure based on detected intent and variables
        config = self.auto_configure(analysis_spec)

        return {
            "summary": {
                "text": f"Comparing {config['variable']} between {len(config['groups'])} groups",
                "test": config['statistical_test'],
                "quick_actions": [
                    "Run with defaults",
                    "Customize settings"
                ]
            },

            "collapsed_settings": {
                # Settings are collapsed but show current values
                "Statistical test": {
                    "value": config['statistical_test'],
                    "auto_selected": True,
                    "click_to_change": True
                },
                "Significance level": {
                    "value": "α = 0.05",
                    "auto_selected": True,
                    "click_to_change": True
                },
                "Missing data": {
                    "value": "Listwise deletion (recommended)",
                    "auto_selected": True,
                    "click_to_change": True
                }
            },

            "progressive_actions": [
                {
                    "label": "⚙️ Show all settings",
                    "action": "expand_all_settings"
                },
                {
                    "label": "📊 Advanced visualizations",
                    "action": "show_viz_options",
                    "badge": "Advanced"
                },
                {
                    "label": "🔍 Sensitivity analysis",
                    "action": "configure_sensitivity",
                    "badge": "Expert"
                }
            ]
        }

# Example rendering
ui = SmartDefaultsUI()
config_ui = ui.render_analysis_config({
    "intent": "COMPARE",
    "variable": "length_of_stay",
    "groups": ["diabetic", "non_diabetic"]
})

# Renders:
# ┌──────────────────────────────────────┐
# │ Comparing length_of_stay between     │
# │ 2 groups                              │
# │                                       │
# │ Test: Mann-Whitney U (auto-selected) │
# │ α = 0.05                              │
# │ Missing data: Listwise deletion      │
# │                                       │
# │ [Run with defaults]  [Customize]     │
# │                                       │
# │ ⚙️ Show all settings                 │
# │ 📊 Advanced visualizations [Advanced]│
# │ 🔍 Sensitivity analysis [Expert]     │
# └──────────────────────────────────────┘
```

### Analytics-Driven Progressive Disclosure

Use analytics to determine what to show by default:

```python
class AdaptiveUI:
    """
    Adjust progressive disclosure based on user behavior
    """

    def __init__(self, user_profile: dict):
        self.user_profile = user_profile
        self.feature_usage = self.load_feature_usage()

    def should_expand_section(self, section: str) -> bool:
        """
        Decide if section should be expanded by default
        """

        # Beginner users: Keep advanced sections collapsed
        if self.user_profile['experience_level'] == 'beginner':
            return section in ['basic_filters', 'simple_options']

        # Frequent users of a feature: Expand it by default
        usage_frequency = self.feature_usage.get(section, 0)
        if usage_frequency > 0.7:  # Used in 70%+ of sessions
            return True

        # Recently used features: Expand
        if self.was_recently_used(section, days=7):
            return True

        return False

    def adapt_suggestions(self, context: dict):
        """
        Adapt suggestions based on user's typical workflows
        """

        # Learn from past queries
        similar_queries = self.find_similar_past_queries(context['query'])

        if similar_queries:
            # Suggest variables/settings user typically uses
            return {
                "suggested_variables": self.extract_common_variables(similar_queries),
                "suggested_settings": self.extract_common_settings(similar_queries),
                "hint": "Based on your previous similar analyses"
            }

        # Fall back to general suggestions
        return self.default_suggestions(context)

# Example usage
ui = AdaptiveUI(user_profile={
    "experience_level": "intermediate",
    "role": "clinical_researcher"
})

# For a power user who frequently adjusts statistical settings
ui.should_expand_section("statistical_options")  # True

# For a beginner
ui.should_expand_section("statistical_options")  # False
```

---

## 7. Session State Management

### Architecture Patterns

#### **In-Memory State (Development)**

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class ConversationMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    intent: Optional[str] = None
    entities: Dict = field(default_factory=dict)

@dataclass
class QueryState:
    """Current query being built"""
    intent: Optional[str] = None
    cohort: Optional[dict] = None
    variables: List[str] = field(default_factory=list)
    time_range: Optional[dict] = None
    statistical_test: Optional[str] = None
    parameters: Dict = field(default_factory=dict)

@dataclass
class SessionState:
    """Complete session state"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime

    # Conversation history
    messages: List[ConversationMessage] = field(default_factory=list)

    # Current query state
    current_query: QueryState = field(default_factory=QueryState)

    # Context across turns
    active_cohort: Optional[dict] = None
    active_variables: List[str] = field(default_factory=list)
    last_result: Optional[dict] = None

    # User preferences (learned during session)
    preferences: Dict = field(default_factory=dict)

class SessionManager:
    """Simple in-memory session management"""

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def create_session(self, user_id: str) -> str:
        import uuid
        session_id = str(uuid.uuid4())

        self.sessions[session_id] = SessionState(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, updates: dict):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = datetime.now()

            for key, value in updates.items():
                setattr(session, key, value)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities: Optional[dict] = None
    ):
        session = self.sessions[session_id]

        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            intent=intent,
            entities=entities or {}
        )

        session.messages.append(message)
        session.last_activity = datetime.now()

# Usage
sm = SessionManager()

# Create session
session_id = sm.create_session(user_id="researcher_123")

# Track conversation
sm.add_message(
    session_id,
    role="user",
    content="Show diabetic patients",
    intent="FILTER",
    entities={"condition": "diabetes"}
)

sm.add_message(
    session_id,
    role="assistant",
    content="Found 247 diabetic patients. What would you like to know?"
)

# Update query state
session = sm.get_session(session_id)
session.active_cohort = {"condition": "diabetes", "count": 247}
session.current_query.cohort = {"condition": "diabetes"}

# Next turn uses context
sm.add_message(
    session_id,
    role="user",
    content="What is their average age?",  # "their" resolved using session.active_cohort
    intent="AGGREGATE"
)
```

#### **Persistent State (Production)**

```python
from redis import Redis
import json
from datetime import timedelta

class RedisSessionManager:
    """Production session management with Redis"""

    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url)
        self.session_ttl = timedelta(hours=24)  # Session expires after 24h inactivity

    def create_session(self, user_id: str) -> str:
        import uuid
        session_id = str(uuid.uuid4())

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "messages": [],
            "current_query": {},
            "context": {}
        }

        # Store in Redis with TTL
        self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            json.dumps(session_data)
        )

        # Add to user's session list
        self.redis.sadd(f"user_sessions:{user_id}", session_id)

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        data = self.redis.get(f"session:{session_id}")
        if data:
            # Refresh TTL on access
            self.redis.expire(f"session:{session_id}", self.session_ttl)
            return json.loads(data)
        return None

    def update_session(self, session_id: str, updates: dict):
        session_data = self.get_session(session_id)
        if session_data:
            session_data.update(updates)
            session_data['last_activity'] = datetime.now().isoformat()

            self.redis.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session_data)
            )

    def add_message(self, session_id: str, message: dict):
        session_data = self.get_session(session_id)
        if session_data:
            session_data['messages'].append(message)
            session_data['last_activity'] = datetime.now().isoformat()

            self.redis.setex(
                f"session:{session_id}",
                self.session_ttl,
                json.dumps(session_data)
            )

    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all active sessions for a user"""
        return [
            s.decode('utf-8')
            for s in self.redis.smembers(f"user_sessions:{user_id}")
        ]

    def cleanup_expired_sessions(self):
        """Remove references to expired sessions"""
        # Redis automatically removes expired keys
        # This cleans up the user_sessions sets

        for key in self.redis.scan_iter("user_sessions:*"):
            user_id = key.decode('utf-8').split(':')[1]
            sessions = self.redis.smembers(key)

            for session_id in sessions:
                # Check if session still exists
                if not self.redis.exists(f"session:{session_id}"):
                    self.redis.srem(key, session_id)
```

#### **Database-Backed State (Enterprise)**

```python
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Session(Base):
    __tablename__ = 'chat_sessions'

    session_id = Column(String(36), primary_key=True)
    user_id = Column(String(255), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    state = Column(JSON)  # Current session state

class Message(Base):
    __tablename__ = 'chat_messages'

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), index=True)
    role = Column(String(20))  # 'user' or 'assistant'
    content = Column(Text)
    intent = Column(String(50), nullable=True)
    entities = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DatabaseSessionManager:
    """Enterprise session management with PostgreSQL"""

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_session(self, user_id: str) -> str:
        import uuid
        session_id = str(uuid.uuid4())

        db = self.SessionLocal()
        try:
            session = Session(
                session_id=session_id,
                user_id=user_id,
                state={
                    "current_query": {},
                    "context": {}
                }
            )
            db.add(session)
            db.commit()
            return session_id
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[dict]:
        db = self.SessionLocal()
        try:
            session = db.query(Session).filter_by(session_id=session_id).first()
            if session:
                return {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "created_at": session.created_at,
                    "last_activity": session.last_activity,
                    "state": session.state
                }
            return None
        finally:
            db.close()

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[dict]:
        """Retrieve conversation history"""
        db = self.SessionLocal()
        try:
            messages = db.query(Message)\
                .filter_by(session_id=session_id)\
                .order_by(Message.timestamp.desc())\
                .limit(limit)\
                .all()

            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "intent": msg.intent,
                    "entities": msg.entities,
                    "timestamp": msg.timestamp
                }
                for msg in reversed(messages)
            ]
        finally:
            db.close()

    def add_message(self, session_id: str, message_data: dict):
        import uuid
        db = self.SessionLocal()
        try:
            message = Message(
                id=str(uuid.uuid4()),
                session_id=session_id,
                **message_data
            )
            db.add(message)

            # Update session last_activity
            session = db.query(Session).filter_by(session_id=session_id).first()
            if session:
                session.last_activity = datetime.utcnow()

            db.commit()
        finally:
            db.close()
```

### State Management Best Practices

From 2025 research on chatbot session management:

#### **1. Multi-Level State Tracking**

```python
# Three levels of state
STATE_LEVELS = {
    "user_state": {
        # Persists across sessions
        "user_id": "researcher_123",
        "preferences": {
            "default_significance_level": 0.05,
            "preferred_visualizations": ["box_plot", "scatter"],
            "typical_cohorts": ["diabetic_patients", "elderly_patients"]
        },
        "expertise_level": "intermediate"
    },

    "conversation_state": {
        # Persists for current conversation
        "conversation_id": "conv_456",
        "active_cohort": {"condition": "diabetes"},
        "active_variables": ["hba1c", "glucose"],
        "last_result": {"test": "t_test", "p_value": 0.023}
    },

    "turn_state": {
        # Only for current turn
        "current_intent": "COMPARE",
        "extracted_entities": {"group1": "treatment_a", "group2": "treatment_b"},
        "clarification_needed": False
    }
}
```

#### **2. Context Window Management**

```python
class ContextWindowManager:
    """
    Manage conversation context for LLM calls
    """

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def build_context_window(
        self,
        session_state: SessionState,
        current_query: str,
        max_history_turns: int = 5
    ) -> str:
        """
        Build context string for LLM, prioritizing recent and relevant history
        """

        context_parts = []

        # 1. System context (always included)
        context_parts.append(self.build_system_context(session_state))

        # 2. Active context (current cohort, variables, etc.)
        if session_state.active_cohort:
            context_parts.append(
                f"Active cohort: {session_state.active_cohort['description']}"
            )

        if session_state.active_variables:
            context_parts.append(
                f"Variables in scope: {', '.join(session_state.active_variables)}"
            )

        # 3. Recent conversation history (sliding window)
        recent_messages = session_state.messages[-max_history_turns:]
        for msg in recent_messages:
            context_parts.append(f"{msg.role}: {msg.content}")

        # 4. Current query
        context_parts.append(f"user: {current_query}")

        # Combine and truncate if needed
        full_context = "\n".join(context_parts)

        if self.estimate_tokens(full_context) > self.max_tokens:
            # Truncate older messages
            return self.smart_truncate(context_parts, self.max_tokens)

        return full_context

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
```

#### **3. State Reconciliation Pattern**

```python
class StateReconciliation:
    """
    Handle state conflicts when multiple updates occur
    """

    def reconcile_query_state(
        self,
        existing_state: QueryState,
        new_input: dict
    ) -> QueryState:
        """
        Merge new input with existing state, resolving conflicts
        """

        # Intent: New intent overrides unless it's "REFINE"
        if new_input.get('intent') == 'REFINE':
            # Keep existing intent
            pass
        elif new_input.get('intent'):
            existing_state.intent = new_input['intent']

        # Cohort: New cohort replaces or refines existing
        if new_input.get('cohort'):
            if new_input['cohort'].get('operation') == 'AND':
                # Combine with existing cohort
                existing_state.cohort = self.combine_cohorts(
                    existing_state.cohort,
                    new_input['cohort']
                )
            else:
                # Replace cohort
                existing_state.cohort = new_input['cohort']

        # Variables: Additive by default
        if new_input.get('variables'):
            existing_vars = set(existing_state.variables)
            new_vars = set(new_input['variables'])
            existing_state.variables = list(existing_vars | new_vars)

        # Time range: New overrides if specified
        if new_input.get('time_range'):
            existing_state.time_range = new_input['time_range']

        return existing_state

# Example
reconciler = StateReconciliation()

# Existing state
state = QueryState(
    intent="COMPARE",
    cohort={"condition": "diabetes"},
    variables=["hba1c", "glucose"]
)

# User adds refinement: "Also look at age"
new_input = {
    "intent": "REFINE",
    "variables": ["age"]
}

reconciled_state = reconciler.reconcile_query_state(state, new_input)
# Result: variables = ["hba1c", "glucose", "age"]
```

---

## 8. Error Handling for Ambiguous Intent

### Disambiguation Patterns

#### **Confidence Threshold Pattern**

```python
class AmbiguityDetector:
    """
    Detect and handle ambiguous intents
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        ambiguity_threshold: float = 0.15
    ):
        self.confidence_threshold = confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold

    def classify_with_ambiguity_detection(
        self,
        query: str
    ) -> dict:
        """
        Classify intent and detect ambiguity
        """

        # Get top-k predictions with confidence scores
        predictions = self.intent_classifier.predict_proba(query, top_k=3)

        top_intent = predictions[0]
        second_intent = predictions[1] if len(predictions) > 1 else None

        # Case 1: High confidence, clear intent
        if top_intent['confidence'] > self.confidence_threshold:
            return {
                "status": "clear",
                "intent": top_intent['label'],
                "confidence": top_intent['confidence']
            }

        # Case 2: Low confidence, fallback triggered
        if top_intent['confidence'] < self.confidence_threshold and (
            not second_intent or
            top_intent['confidence'] - second_intent['confidence'] > self.ambiguity_threshold
        ):
            return {
                "status": "low_confidence",
                "intent": top_intent['label'],
                "confidence": top_intent['confidence'],
                "action": "request_clarification"
            }

        # Case 3: Ambiguous - multiple intents with similar confidence
        if second_intent:
            confidence_diff = top_intent['confidence'] - second_intent['confidence']

            if confidence_diff < self.ambiguity_threshold:
                return {
                    "status": "ambiguous",
                    "candidates": [top_intent, second_intent],
                    "action": "disambiguate"
                }

        return {
            "status": "clear",
            "intent": top_intent['label'],
            "confidence": top_intent['confidence']
        }

# Example usage
detector = AmbiguityDetector(
    confidence_threshold=0.85,
    ambiguity_threshold=0.15
)

# Clear intent
result = detector.classify_with_ambiguity_detection(
    "Compare mortality rates between treatment groups"
)
# {'status': 'clear', 'intent': 'COMPARE', 'confidence': 0.95}

# Ambiguous intent
result = detector.classify_with_ambiguity_detection(
    "Look at diabetes and outcomes"
)
# {
#   'status': 'ambiguous',
#   'candidates': [
#     {'label': 'DESCRIBE', 'confidence': 0.52},
#     {'label': 'CORRELATE', 'confidence': 0.48}
#   ],
#   'action': 'disambiguate'
# }
```

#### **Two-Stage Fallback Pattern**

From 2025 research: "Two-stage fallback lets users clarify themselves by suggesting possible actions twice before heading to ultimate fallback."

```python
class TwoStageFallback:
    """
    Implement two-stage fallback for ambiguous queries
    """

    def __init__(self):
        self.fallback_attempts = {}  # Track attempts per session

    def handle_ambiguous_query(
        self,
        session_id: str,
        query: str,
        ambiguity_result: dict
    ) -> dict:
        """
        Two-stage clarification before ultimate fallback
        """

        # Track attempts
        attempts = self.fallback_attempts.get(session_id, 0)

        if attempts == 0:
            # Stage 1: Suggest specific interpretations
            return self.stage1_specific_suggestions(ambiguity_result)

        elif attempts == 1:
            # Stage 2: Broader suggestions with examples
            return self.stage2_broad_suggestions(query)

        else:
            # Ultimate fallback: Human handoff or structured input
            return self.ultimate_fallback()

    def stage1_specific_suggestions(self, ambiguity_result: dict) -> dict:
        """
        First stage: Offer specific interpretations
        """
        candidates = ambiguity_result['candidates']

        return {
            "message": "I'm not quite sure what you're looking for. Did you want to:",
            "suggestions": [
                {
                    "intent": c['label'],
                    "description": self.get_intent_description(c['label']),
                    "example": self.get_example_for_intent(c['label'])
                }
                for c in candidates
            ],
            "stage": 1,
            "allow_rephrase": True,
            "rephrase_prompt": "Or rephrase your question"
        }

    def stage2_broad_suggestions(self, original_query: str) -> dict:
        """
        Second stage: Broader suggestions with categories
        """
        return {
            "message": "Let me help you phrase your question. What type of analysis?",
            "categories": [
                {
                    "category": "Comparison",
                    "examples": [
                        "Compare [outcome] between [group A] and [group B]",
                        "Are [group A] different from [group B] in terms of [variable]?"
                    ]
                },
                {
                    "category": "Prediction",
                    "examples": [
                        "Predict [outcome] based on [variables]",
                        "What factors influence [outcome]?"
                    ]
                },
                {
                    "category": "Correlation",
                    "examples": [
                        "Is [variable A] related to [variable B]?",
                        "Show relationship between [variable A] and [variable B]"
                    ]
                },
                {
                    "category": "Description",
                    "examples": [
                        "Describe [variable] in [population]",
                        "What is the distribution of [variable]?"
                    ]
                }
            ],
            "stage": 2,
            "offer_wizard": True,
            "wizard_prompt": "Or use the guided query builder"
        }

    def ultimate_fallback(self) -> dict:
        """
        Final fallback: Structured input or human assistance
        """
        return {
            "message": "I'm having trouble understanding. Let me help you differently.",
            "options": [
                {
                    "type": "wizard",
                    "label": "Use guided query builder",
                    "description": "I'll ask you step-by-step questions"
                },
                {
                    "type": "template",
                    "label": "Start from a template",
                    "description": "Choose a common analysis template"
                },
                {
                    "type": "contact",
                    "label": "Contact data analyst",
                    "description": "Get help from a human expert"
                }
            ],
            "stage": "ultimate"
        }

# Example interaction
fallback = TwoStageFallback()

# First attempt
response = fallback.handle_ambiguous_query(
    session_id="sess_123",
    query="Look at diabetes patients",
    ambiguity_result={
        'candidates': [
            {'label': 'DESCRIBE', 'confidence': 0.52},
            {'label': 'FILTER', 'confidence': 0.48}
        ]
    }
)
# Returns stage 1 suggestions

# User still unclear, second attempt
fallback.fallback_attempts["sess_123"] = 1
response = fallback.handle_ambiguous_query(
    session_id="sess_123",
    query="I want to see diabetes data",
    ambiguity_result={...}
)
# Returns stage 2 with examples and categories

# Third attempt - ultimate fallback
fallback.fallback_attempts["sess_123"] = 2
response = fallback.handle_ambiguous_query(
    session_id="sess_123",
    query="diabetes",
    ambiguity_result={...}
)
# Returns wizard, template, or human contact options
```

#### **Context-Aware Disambiguation**

From research: "Low-accuracy models cannot handle ambiguous queries, but utilizing additional information such as contextual data in the right manner can help identify intent."

```python
class ContextAwareDisambiguation:
    """
    Use conversation context to resolve ambiguity
    """

    def disambiguate_with_context(
        self,
        query: str,
        ambiguity_result: dict,
        session_state: SessionState
    ) -> dict:
        """
        Use session context to resolve ambiguous intent
        """

        candidates = ambiguity_result['candidates']

        # Score each candidate based on context
        context_scores = []

        for candidate in candidates:
            score = self.score_candidate_with_context(
                candidate,
                session_state
            )
            context_scores.append({
                **candidate,
                "context_score": score
            })

        # Re-rank by context score
        context_scores.sort(key=lambda x: x['context_score'], reverse=True)

        # If context strongly suggests one intent, use it
        if context_scores[0]['context_score'] > 0.8:
            return {
                "status": "resolved_by_context",
                "intent": context_scores[0]['label'],
                "confidence": context_scores[0]['confidence'],
                "context_confidence": context_scores[0]['context_score']
            }

        # Otherwise, still ambiguous
        return {
            "status": "ambiguous",
            "candidates": context_scores[:2]
        }

    def score_candidate_with_context(
        self,
        candidate: dict,
        session_state: SessionState
    ) -> float:
        """
        Score how well candidate intent fits with session context
        """
        score = 0.5  # Base score

        # Check recent intent patterns
        recent_intents = [
            msg.intent for msg in session_state.messages[-3:]
            if msg.intent
        ]

        if recent_intents:
            # If user has been doing comparisons, COMPARE is more likely
            if candidate['label'] == 'COMPARE' and 'COMPARE' in recent_intents:
                score += 0.2

            # If switching contexts, lower score for same intent
            if all(i == recent_intents[0] for i in recent_intents):
                if candidate['label'] == recent_intents[0]:
                    score -= 0.1  # Might be switching tasks

        # Check if intent makes sense with active cohort
        if session_state.active_cohort:
            if candidate['label'] in ['COMPARE', 'AGGREGATE']:
                score += 0.1  # These work well with active cohorts

        # Check if variables in scope suggest certain intents
        if session_state.active_variables:
            if len(session_state.active_variables) >= 2:
                if candidate['label'] == 'CORRELATE':
                    score += 0.15  # Multiple variables suggest correlation

        return min(score, 1.0)

# Example
disambiguator = ContextAwareDisambiguation()

# Ambiguous query: "Show the relationship"
# But context suggests comparison is more likely
session_state = SessionState(
    session_id="sess_123",
    user_id="user_456",
    messages=[
        ConversationMessage(
            role="user",
            content="Compare treatment groups",
            intent="COMPARE",
            timestamp=datetime.now()
        ),
        ConversationMessage(
            role="user",
            content="Now for another variable",
            intent="COMPARE",
            timestamp=datetime.now()
        )
    ],
    active_cohort={"treatment_groups": ["A", "B"]}
)

result = disambiguator.disambiguate_with_context(
    query="Show the relationship",
    ambiguity_result={
        'candidates': [
            {'label': 'CORRELATE', 'confidence': 0.51},
            {'label': 'COMPARE', 'confidence': 0.49}
        ]
    },
    session_state=session_state
)

# Result: COMPARE intent selected based on context
# (user has been comparing, has active treatment groups)
```

#### **Clarification Prompt Templates**

```python
CLARIFICATION_PROMPTS = {
    "COMPARE_vs_CORRELATE": {
        "message": "Would you like to:",
        "options": [
            {
                "label": "Compare groups",
                "description": "Test if [outcome] differs between [group A] and [group B]",
                "example": "e.g., Do diabetic patients have higher readmission rates?"
            },
            {
                "label": "Find correlations",
                "description": "Measure relationship between two variables",
                "example": "e.g., Is age associated with length of stay?"
            }
        ]
    },

    "DESCRIBE_vs_AGGREGATE": {
        "message": "What would you like to see:",
        "options": [
            {
                "label": "Summary statistics",
                "description": "Mean, median, distribution of a variable",
                "example": "e.g., Average age: 62 years (SD: 11)"
            },
            {
                "label": "Count/Total",
                "description": "Number or sum of cases",
                "example": "e.g., Total: 247 patients"
            }
        ]
    },

    "PREDICT_vs_CORRELATE": {
        "message": "Are you trying to:",
        "options": [
            {
                "label": "Build a prediction model",
                "description": "Use multiple factors to predict an outcome",
                "example": "e.g., Predict readmission risk using age, comorbidities, etc."
            },
            {
                "label": "Explore relationships",
                "description": "See if two variables are related",
                "example": "e.g., Is BMI related to complications?"
            }
        ]
    }
}

def generate_clarification_prompt(
    top_candidates: list
) -> dict:
    """
    Generate appropriate clarification based on ambiguous intents
    """

    # Get intent pair
    intent_pair = f"{top_candidates[0]['label']}_vs_{top_candidates[1]['label']}"

    # Use template if available
    if intent_pair in CLARIFICATION_PROMPTS:
        return CLARIFICATION_PROMPTS[intent_pair]

    # Generic clarification
    return {
        "message": "I'm not quite sure what you're looking for. Did you want to:",
        "options": [
            {
                "label": c['label'],
                "confidence": c['confidence']
            }
            for c in top_candidates
        ]
    }
```

### Error Recovery Patterns

#### **Graceful Degradation**

```python
class GracefulDegradation:
    """
    Provide partial results when full analysis isn't possible
    """

    def attempt_analysis(
        self,
        query_spec: dict
    ) -> dict:
        """
        Try to provide useful results even if full intent unclear
        """

        try:
            # Try full analysis
            return self.run_full_analysis(query_spec)

        except AmbiguousIntentError as e:
            # Provide what we can determine
            return {
                "status": "partial",
                "message": "I couldn't fully understand your question, but here's what I found:",
                "partial_results": {
                    "cohort_summary": self.get_cohort_summary(query_spec),
                    "available_variables": self.get_relevant_variables(query_spec),
                    "suggested_analyses": self.suggest_analyses(query_spec)
                },
                "clarification_needed": str(e)
            }

        except InsufficientInformationError as e:
            # Ask for missing information
            return {
                "status": "incomplete",
                "message": f"To complete this analysis, I need to know: {e.missing_info}",
                "follow_up_questions": e.questions
            }

# Example
degrader = GracefulDegradation()

# Vague query
result = degrader.attempt_analysis({
    "query": "diabetes patients",
    "intent": "UNKNOWN"
})

# Returns:
# {
#   "status": "partial",
#   "message": "I couldn't fully understand...",
#   "partial_results": {
#     "cohort_summary": "Found 247 patients with diabetes diagnosis",
#     "available_variables": ["hba1c", "glucose", "medications", ...],
#     "suggested_analyses": [
#       "Compare outcomes to non-diabetic patients",
#       "Analyze diabetes medication patterns",
#       "Examine comorbidity rates"
#     ]
#   }
# }
```

---

## Implementation Recommendations

### For MVP (Weeks 1-4)

1. **Start with rule-based + embeddings hybrid**
   - Fast to implement
   - Covers 70-80% of queries
   - No training infrastructure needed

2. **Simple session management**
   - In-memory state for development
   - Redis for production

3. **Basic progressive disclosure**
   - Start with 3 intent types: COMPARE, DESCRIBE, CORRELATE
   - Single-step refinement (no multi-turn wizard)

4. **Simple error handling**
   - Confidence threshold with fallback to clarification
   - Template-based responses

### For Production (Months 2-3)

1. **Fine-tune BERT for intent classification**
   - Use ClinicalBERT as base model
   - Collect 500-1000 labeled examples
   - Aim for 90%+ accuracy

2. **Implement semantic layer**
   - Map natural language to semantic model (like Looker's approach)
   - Reduces errors by 60%+

3. **Multi-turn conversation support**
   - Context retention across turns
   - Reference resolution ("their", "it", "them")
   - Follow-up suggestions

4. **Advanced error handling**
   - Two-stage fallback
   - Context-aware disambiguation
   - Graceful degradation

### For Scale (Months 4-6)

1. **LLM-based reasoning**
   - For complex multi-step analyses
   - Statistical test selection
   - Natural language result explanations

2. **Adaptive UI**
   - Learn from user behavior
   - Personalized suggestions
   - Smart defaults

3. **Multi-agent architecture**
   - Specialized agents for different analysis types
   - Parallel processing for complex queries

---

## Sources

This research draws from the following authoritative sources:

### Natural Language Query Tools & Platforms

- [Natural Language Visualization and the Future of Data Analysis](https://towardsdatascience.com/natural-language-visualization-and-the-future-of-data-analysis-and-presentation/)
- [Best Natural Language Query Tools for Data Analysis (November 2025)](https://index.app/blog/natural-language-query-tools)
- [Natural Language Querying: The Practical Guide for Data Teams in 2025](https://www.symbolicdata.org/natural-language-querying/)
- [Natural Language Query (NLQ): Simplifying Data Access | LANSA](https://lansa.com/blog/business-intelligence/nlq-natural-language-query/)

### Tableau Ask Data

- [Preparing data for natural language interaction in Ask Data](https://www.tableau.com/learn/whitepapers/preparing-data-nlp-in-ask-data)
- [Ask Data: Simplifying analytics with natural language](https://www.tableau.com/blog/ask-data-simplifying-analytics-natural-language-98655)
- ["Ask Data" – natural language queries in Tableau](https://datavis.blog/2019/02/25/tableau-ask-data/)

### Looker Conversational Analytics

- [Conversational Analytics in Looker overview | Google Cloud](https://docs.cloud.google.com/looker/docs/conversational-analytics-overview)
- [Understanding Looker's Conversational Analytics API](https://cloud.google.com/blog/products/data-analytics/understanding-lookers-conversational-analytics-api)
- [How Twilio generated SQL using Looker Modeling Language](https://aws.amazon.com/blogs/machine-learning/how-twilio-generated-sql-using-looker-modeling-language-data-with-amazon-bedrock/)
- [Looker Explore Assistant (Open Source)](https://github.com/looker-open-source/looker-explore-assistant)

### ThoughtSpot

- [Inside Spotter – ThoughtSpot's AI-Powered NLQ Engine](https://training.thoughtspot.com/inside-spotter-thoughtspots-ai-powered-nlq-engine)
- [Natural Language Question Answering Systems - ThoughtSpot Patent](https://www.freepatentsonline.com/y2019/0272296.html)

### Intent Classification

- [Intent Classification: 2025 Techniques for NLP Models](https://labelyourdata.com/articles/machine-learning/intent-classification)
- [How intent classification works in NLU](https://botfront.io/blog/how-intent-classification-works-in-nlu/)
- [Rasa NLU in Depth: Intent Classification](https://rasa.com/blog/rasa-nlu-in-depth-part-1-intent-classification/)
- [Using LLMs for Intent Classification | Rasa](https://legacy-docs-oss.rasa.com/docs/rasa/next/llms/llm-intent/)

### Conversational UI & Progressive Disclosure

- [Progressive Disclosure | Interaction Design Foundation](https://www.interaction-design.org/literature/topics/progressive-disclosure)
- [Progressive Disclosure - Nielsen Norman Group](https://www.nngroup.com/articles/progressive-disclosure/)
- [Progressive Disclosure UI Patterns - Agentic Design](https://agentic-design.ai/patterns/ui-ux-patterns/progressive-disclosure-patterns)
- [Progressive Disclosure Examples to Simplify Complex SaaS](https://userpilot.com/blog/progressive-disclosure-examples/)

### Session Management

- [AI Chatbot Session Management: Best Practices](https://optiblack.com/insights/ai-chatbot-session-management-best-practices)
- [Optimize Session Management for AI Conversation Apps](https://medium.com/@aslam.develop912/master-session-management-for-ai-apps-a-practical-guide-with-backend-frontend-code-examples-cb36c676ea77)
- [Amazon Bedrock Session Management APIs](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-launches-session-management-apis-for-generative-ai-applications-preview/)
- [Introduction to Conversational Context | Google ADK](https://google.github.io/adk-docs/sessions/)

### Error Handling & Disambiguation

- [Handling chatbot failure gracefully | Towards Data Science](https://towardsdatascience.com/handling-chatbot-failure-gracefully-466f0fb1dcc5/)
- [NLU Disambiguation - What to do when the NLU is not sure](https://www.engati.com/blog/nlu-disambiguation)
- [Error Correction in Conversational AI: A Review](https://www.mdpi.com/2673-2688/5/2/41)
- [Chatbot Error Handling - Managing Mistakes](https://moldstud.com/articles/p-chatbot-error-handling-managing-mistakes-and-improving-accuracy-in-ai-responses)

### Statistical Test Selection

- [Evaluating LLMs for selection of statistical test (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11584160/)
- [Clinical NLP survey in the UK 2007-2022](https://www.nature.com/articles/s41746-022-00730-6)

### Technical Documentation

- [LangChain Python Documentation](https://docs.langchain.com/oss/python/langchain/)
- [HuggingFace Transformers Documentation](https://github.com/huggingface/transformers)
- [Sentence Transformers](https://github.com/huggingface/sentence-transformers)

---

## Appendix: Code Examples Repository

Complete working examples are available at:
- Intent classification models: `/models/intent_classifiers/`
- Session management implementations: `/session_management/`
- UI components: `/frontend/components/`
- Statistical test mapping: `/analytics/test_selection/`

---

**Document Version:** 1.0
**Last Updated:** 2025-12-24
**Research conducted for:** MD Data Explorer clinical analytics platform
