# Dataset Structure Patterns

**Date:** 2025-12-24  
**Purpose:** Document the consistent file structure patterns across datasets

---

## 📁 Standard Dataset Structure

Clinical datasets from PhysioNet and similar sources follow a consistent structure that includes:

1. **Data Files** - The actual dataset (CSV, PSV, or relational tables) in `data/raw/`
2. **README/Documentation** - Data dictionary with column descriptions
   - Original location: In `data/raw/{dataset}/` alongside data files
   - **Centralized location:** `data/dictionaries/` - PDF documentation files for all datasets
3. **License Files** - Usage terms and conditions
4. **Checksums** - File integrity verification

**Note:** Data dictionaries are available in two locations:
- **Original location:** In the raw data directory alongside data files (README.txt, README.md, or PDF)
- **Centralized location:** `data/dictionaries/` - Contains PDF copies for easy reference and NL query implementation

---

## 🔍 COVID-MS Dataset Structure

**Location:** `data/raw/covid_ms/`

```
covid_ms/
├── GDSI_OpenDataset_Final.csv    # Main data file (1,141 patients)
├── README.txt                     # Data dictionary & description
├── LICENSE.txt                     # CC BY 4.0 License
└── SHA256SUMS.txt                 # File integrity checksums
```

### README.txt Contents
- Dataset description
- Repository contents
- Usage guidelines
- License information
- Data collection methodology

**Example:**
```
Patient-level dataset to study the effect of COVID-19 in people with Multiple Sclerosis

Description:
This repository contains the anonymized open-sourced dataset from the 
COVID-19 and Multiple Sclerosis (MS) Global Data Sharing Initiative (GDSI).
The dataset comprises data entered by people with MS and clinicians...

Repository Contents:
- 'GDSI_OpenDataset_Final.csv': The anonymized and de-identified dataset 
  containing data collected from 1141 people with Multiple Sclerosis...
```

---

## 🔍 Sepsis Dataset Structure

**Location:** `data/raw/sepsis/physionet.org/files/challenge-2019/1.0.0/`

```
sepsis/
├── training/                       # Training data directory
│   ├── p000001.psv                # Patient data files (PSV format)
│   ├── p000002.psv
│   └── ...                        # Multiple PSV files (one per patient)
├── physionet_challenge_2019_ccm_manuscript.pdf  # Data dictionary & methodology
├── LICENSE.txt                     # ODbL (Open Database License)
└── SHA256SUMS.txt                  # File integrity checksums
```

### Sepsis Documentation Structure

**Unlike COVID-MS which uses README.txt, Sepsis uses a PDF manuscript:**

- **`physionet_challenge_2019_ccm_manuscript.pdf`** - Comprehensive documentation including:
  - Dataset description and methodology
  - Data collection process
  - Column/variable definitions
  - Data format specifications (PSV format)
  - Outcome definitions (SepsisLabel)
  - Feature descriptions (vital signs, lab values, etc.)
  - Usage guidelines and challenge details

**Key Information in PDF:**
- **Data Format**: PSV (pipe-separated values)
- **Time Series**: Hourly measurements per patient
- **Outcome**: `SepsisLabel` (binary: 0/1)
- **Features**: Age, Gender, vital signs, lab values, etc.
- **Aggregation**: Time-series data aggregated to patient-level

**Note:** The PDF serves the same purpose as README.txt - it's the data dictionary and methodology documentation, just in PDF format instead of text.

---

## 🔍 MIMIC Dataset Structure

**MIMIC-III and MIMIC-IV follow the same pattern:**

### MIMIC-IV Demo Structure

```
mimic4_demo/
├── hosp/                          # Hospital module
│   ├── admissions.csv             # Admission records
│   ├── patients.csv               # Patient demographics
│   ├── diagnoses_icd.csv          # ICD diagnosis codes
│   ├── labevents.csv              # Laboratory results
│   └── ...
├── icu/                           # ICU module
│   ├── icustays.csv
│   ├── chartevents.csv
│   └── ...
├── README.md                      # Comprehensive data dictionary
├── LICENSE.txt                    # Usage license
└── documentation/                 # Additional documentation
    ├── MIMIC_IV_Data_Dictionary.pdf
    └── ...
```

### MIMIC README Contents

**MIMIC README files include:**
- **Table descriptions** - What each table contains
- **Column definitions** - Detailed field descriptions
- **Data types** - Column types and formats
- **Relationships** - Foreign key relationships between tables
- **Usage examples** - Sample queries and workflows
- **Data collection methodology** - How data was collected
- **Temporal relationships** - How time is represented

**Example MIMIC README structure:**
```markdown
# MIMIC-IV Data Dictionary

## Table: admissions

| Column | Type | Description |
|--------|------|-------------|
| hadm_id | integer | Hospital admission ID |
| subject_id | integer | Patient identifier |
| admittime | timestamp | Admission time |
| dischtime | timestamp | Discharge time |
| ...

## Relationships
- admissions.subject_id → patients.subject_id
- admissions.hadm_id → diagnoses_icd.hadm_id
```

---

## ✅ Why This Pattern Matters

### 1. **Automatic Configuration Generation**

The consistent structure enables:
- **Data dictionary parsing** - Extract column descriptions from README
- **Schema inference** - Understand data types and relationships
- **Semantic layer auto-config** - Generate `datasets.yaml` entries from documentation

### 2. **Semantic Understanding for NL Queries**

When implementing NL queries (Phase 3), the data dictionary provides:
- **Variable synonyms** - "mortality" = "death" = "deceased"
- **Column descriptions** - Understand what each field means
- **Relationships** - Know which tables join together
- **Context for RAG** - Use README content as knowledge base

### 3. **Consistent User Experience**

Users working with different datasets see:
- Same file structure
- Same documentation format
- Same data dictionary style
- Familiar patterns across datasets

---

## 🔄 Platform Integration

### Centralized Data Dictionary Location

**All data dictionaries are now centralized in:**
- `data/dictionaries/` - Contains PDF documentation files for all datasets

**Current dictionaries:**
- `Patient-level dataset to study the effect of COVID-19 in people with Multiple Sclerosis v1.0.1.pdf` - COVID-MS
- `Early Prediction of Sepsis from Clinical Data_ The PhysioNet_Computing in Cardiology Challenge 2019 v1.0.0.pdf` - Sepsis
- `Microbiological, Immunological and Biochemical Characteristics of the Development of Ventilator Associated Pneumonia v1.1.0.pdf` - VAP dataset

**Benefits of centralization:**
- Single location for all data dictionary documentation
- Easier to reference for NL query implementation
- Consistent access pattern across datasets
- Can be used for RAG context generation

### Current Implementation

**COVID-MS:**
- ✅ Data file loaded from `data/raw/covid_ms/GDSI_OpenDataset_Final.csv`
- ✅ README.txt available in `data/raw/covid_ms/README.txt`
- ✅ PDF dictionary available in `data/dictionaries/`
- ✅ Semantic layer config manually created from README

**Sepsis:**
- ✅ Data files loaded from PSV files in `data/raw/sepsis/.../training/` directory
- ✅ PDF manuscript available in `data/dictionaries/` (centralized)
- ✅ Original PDF also in `data/raw/sepsis/.../physionet_challenge_2019_ccm_manuscript.pdf`
- ✅ Semantic layer config manually created from PDF documentation
- ✅ Time-series aggregation to patient-level

**MIMIC (Planned):**
- ⏳ Data files loaded from relational CSVs
- ⏳ README.md available for reference
- ⏳ PDF dictionary can be added to `data/dictionaries/`
- ⏳ Semantic layer config to be created from data dictionary

### Future Enhancement: Auto-Parsing

**Goal:** Automatically generate semantic layer config from data dictionaries

```python
# Future: Auto-generate config from data dictionary
# Can use centralized location: data/dictionaries/

def parse_data_dictionary(dict_path: str) -> dict:
    """
    Parse data dictionary to extract:
    - Table/column descriptions
    - Data types
    - Relationships
    - Outcome definitions
    
    Supports:
    - README.md or README.txt (text/markdown)
    - PDF files (from data/dictionaries/ or raw locations)
    """
    if dict_path.endswith('.pdf'):
        return parse_pdf_dictionary(dict_path)
    elif dict_path.endswith(('.txt', '.md')):
        return parse_text_dictionary(dict_path)
    else:
        raise ValueError(f"Unsupported dictionary format: {dict_path}")

# Example usage with centralized dictionaries
covid_dict = parse_data_dictionary("data/dictionaries/Patient-level dataset to study the effect of COVID-19 in people with Multiple Sclerosis v1.0.1.pdf")
sepsis_dict = parse_data_dictionary("data/dictionaries/Early Prediction of Sepsis from Clinical Data_ The PhysioNet_Computing in Cardiology Challenge 2019 v1.0.0.pdf")
```

**Benefits:**
- Zero-code dataset addition (already achieved)
- Zero-config dataset addition (future)
- Automatic variable name mapping
- Relationship inference from documentation
- Centralized dictionary location simplifies access

---

## 📊 Comparison: COVID-MS vs. Sepsis vs. MIMIC

| Aspect | COVID-MS | Sepsis | MIMIC |
|--------|----------|--------|-------|
| **Data Format** | Single CSV | Multiple PSV files (time-series) | Multiple relational CSVs |
| **Documentation** | README.txt | PDF manuscript | README.md + PDF |
| **Structure** | Flat table | Time-series → aggregated | Normalized relational |
| **Data Dictionary** | In README.txt | In PDF manuscript | Comprehensive in README.md |
| **License** | CC BY 4.0 | ODbL (Open Database License) | PhysioNet Credentialed Health Data |
| **Checksums** | SHA256SUMS.txt | SHA256SUMS.txt | Included in distribution |

**Common Patterns:**
- ✅ All include documentation with data dictionary (README.txt, PDF, or README.md)
- ✅ All include license files
- ✅ All include file integrity checksums (SHA256SUMS.txt)
- ✅ All document column meanings and data types
- ✅ All provide usage guidelines
- ✅ All follow PhysioNet distribution patterns

---

## 🎯 For NL Query Implementation

When implementing semantic NL queries (Phase 3), the data dictionary becomes critical:

### 1. **Entity Extraction**

```python
# Use documentation to understand variable names
# Centralized dictionaries location: data/dictionaries/

# For COVID-MS: PDF from centralized location
data_dictionary = parse_pdf("data/dictionaries/Patient-level dataset to study the effect of COVID-19 in people with Multiple Sclerosis v1.0.1.pdf")
# Or use README.txt from raw data
data_dictionary = parse_readme("data/raw/covid_ms/README.txt")

# For Sepsis: PDF from centralized location
data_dictionary = parse_pdf("data/dictionaries/Early Prediction of Sepsis from Clinical Data_ The PhysioNet_Computing in Cardiology Challenge 2019 v1.0.0.pdf")

# For MIMIC: README.md (when available)
data_dictionary = parse_readme("data/raw/mimic4_demo/README.md")
# Or PDF from centralized location when added

# User query: "What's the mortality rate?"
# Match "mortality" to:
# - Documentation description: "death" → deathtime column
# - Documentation description: "mortality" → dod (date of death) column
# - Sepsis: "sepsis" → SepsisLabel column
```

### 2. **RAG Context**

```python
# Use documentation content as RAG knowledge base
# Works with README.txt, PDF, or README.md

semantic_context = {
    'data_dictionary': documentation_content,  # From README/PDF
    'table_descriptions': extract_table_descriptions(doc),
    'column_meanings': extract_column_descriptions(doc),
    'relationships': extract_relationships(doc),
    'outcome_definitions': extract_outcome_definitions(doc)  # SepsisLabel, mortality, etc.
}

# LLM uses this context to understand user queries
# Works across all dataset types (CSV, PSV, relational)
```

### 3. **Synonym Mapping**

```python
# Extract synonyms from README descriptions
# "mortality" → ["death", "deceased", "died", "fatal"]
# "admission" → ["hospitalization", "stay", "encounter"]
```

---

## 📝 Documentation Standards

### Recommended Documentation Structure

For consistency, dataset documentation (README.txt, README.md, or PDF) should include:

1. **Dataset Overview**
   - Description
   - Source
   - Collection methodology

2. **Data Dictionary**
   - Table/column descriptions
   - Data types
   - Value ranges/domains
   - Outcome definitions

3. **Relationships** (for relational data)
   - Foreign keys
   - Join patterns
   - Temporal relationships

4. **Usage Guidelines**
   - How to load data
   - Common queries
   - Analysis examples
   - Data format specifications (CSV, PSV, etc.)

5. **License & Citation**
   - License terms
   - Citation requirements
   - Attribution

**Note:** Format can vary (text, markdown, PDF) but content should cover these areas.

---

## ✅ Checklist for New Datasets

When adding a new dataset, ensure it includes:

- [ ] Data files (CSV/PSV/relational)
- [ ] Documentation with data dictionary (README.txt, README.md, or PDF)
- [ ] License file
- [ ] File integrity checksums (optional but recommended)
- [ ] Column/variable descriptions
- [ ] Data type documentation
- [ ] Outcome definitions
- [ ] Relationship documentation (for relational data)
- [ ] Usage guidelines

**Documentation Format Examples:**
- **COVID-MS**: README.txt (text file)
- **Sepsis**: PDF manuscript (comprehensive documentation)
- **MIMIC**: README.md + PDF (markdown + detailed PDF)

---

**This consistent structure enables the platform to automatically understand and integrate new datasets, supporting both current config-driven architecture and future NL query capabilities.**

