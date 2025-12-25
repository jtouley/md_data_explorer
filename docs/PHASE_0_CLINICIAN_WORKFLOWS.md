# Phase 0: Clinician-Centered Workflows

**Version:** 1.0
**Date:** 2025-12-24
**Goal:** Make the platform usable by clinical researchers with no programming experience
**Priority:** MUST complete before Phase 1 (security fixes can run in parallel)

---

## Executive Summary

The current implementation is **engineer-focused** but the end users are **clinical researchers** who need to:
- Upload their own data without editing YAML files
- Run clinical analyses (not just logistic regression)
- Export publication-ready results for papers/posters
- Get guidance on appropriate statistical tests

**Phase 0 closes the gap between engineering excellence and clinical usability.**

---

## Critical Friction Points in Current UI

### 🔴 **Blocker Issues** (Prevents Clinical Use)

#### 1. No Self-Service Data Upload
**Problem:** Doctors must ask engineers to add datasets via YAML configuration
```yaml
# Current workflow (UNACCEPTABLE for clinicians):
# 1. Doctor emails engineer: "I have patient data in Excel"
# 2. Engineer creates dataset class in Python
# 3. Engineer writes YAML config
# 4. Engineer commits to git
# 5. Doctor can finally use data (days later)
```

**Impact:** Platform is unusable without engineering support for every new dataset

**Solution:** Drag-and-drop data upload with automatic schema detection

---

#### 2. Limited Analysis Options
**Problem:** Only logistic regression available, but survival analysis code exists unused!

**Current:**
- ✅ Logistic regression (for binary outcomes)
- ❌ Survival analysis (Kaplan-Meier, Cox regression) - **exists but not in UI**
- ❌ Descriptive statistics tables
- ❌ Group comparisons (t-test, chi-square)
- ❌ Subgroup analysis
- ❌ Correlation analysis

**Impact:** Doctors forced to export data and use other tools (defeats purpose)

**Solution:** Guided analysis wizard with all clinical analysis types

---

#### 3. Exports Not Publication-Ready
**Problem:** Downloads are raw CSV files, not formatted for journals/posters

**Current Exports:**
```csv
variable,coef,std_err,z,p_value
age,0.043,0.012,3.58,0.00034
sex_M,1.23,0.45,2.73,0.0063
```

**What Doctors Actually Need:**
```
Table 1. Logistic Regression Results (N=247)

Variable             OR (95% CI)      p-value
─────────────────────────────────────────────
Age (years)          1.04 (1.02-1.07) <0.001***
Sex
  Female (ref)       1.00
  Male               3.42 (1.41-8.29)  0.006**

───────────────────────────────────────────────
Model: χ²(2)=18.45, p<0.001, Pseudo R²=0.142
*p<0.05, **p<0.01, ***p<0.001
```

**Impact:** Doctors must manually reformat all results (hours of work)

**Solution:** One-click export to Word tables, PowerPoint slides, publication-ready figures

---

### 🟡 **High-Friction Issues** (Usability Barriers)

#### 4. Technical Terminology Throughout
**Problem:** UI uses engineering terms, not clinical language

| Current (Technical) | Should Be (Clinical) |
|-------------------|---------------------|
| "Semantic Layer" | "Data Analysis" |
| "Metrics & Dimensions" | "What to Measure & How to Group" |
| "Query Builder" | "Custom Analysis" |
| "Unified Schema" | "Standardized Format" |
| "Cohort" | "Patient Group" |
| "Predictor Variables" | "Risk Factors" or "Exposures" |

**Impact:** Cognitive load, intimidating interface, steep learning curve

**Solution:** Clinical terminology layer with tooltips

---

#### 5. No Guided Workflow
**Problem:** Assumes users know what analysis to run

**Current:** "Select variables, click Run Regression"

**What Doctors Need:**
```
"I want to..."
  ○ Compare outcomes between two groups
  ○ Identify risk factors for an outcome
  ○ Analyze survival/time-to-event
  ○ Describe my patient population
  ○ Explore relationships between variables

[System suggests appropriate test based on data types]
```

**Impact:** Doctors may run inappropriate analyses or give up

**Solution:** Analysis wizard that guides users to correct statistical tests

---

#### 6. No Visualizations
**Problem:** No charts, graphs, or clinical plots

**Missing but Essential:**
- Kaplan-Meier survival curves
- Forest plots (for multiple analyses)
- ROC curves (model performance)
- Distribution histograms
- Box plots for group comparisons
- Scatter plots with regression lines

**Impact:** Can't create figures for papers/presentations

**Solution:** Auto-generate publication-quality figures for each analysis type

---

#### 7. No Result Interpretation Help
**Problem:** Shows statistics without explanation

**Current:** "p=0.0034"

**What Doctors Need:**
```
✓ Statistically significant (p=0.003)
  This result is unlikely to be due to chance.

  Interpretation:
  Age is significantly associated with outcome.
  For each additional year of age, the odds of
  the outcome increase by 4% (OR=1.04, 95% CI: 1.02-1.07).
```

**Impact:** Misinterpretation of results, incorrect conclusions

**Solution:** Plain-language interpretation for each statistical result

---

### 🟢 **Polish Issues** (Nice to Have)

#### 8. No Variable Management
- Can't rename variables to readable names ("sex_male_01" → "Male")
- Can't exclude variables from analysis
- Can't create derived variables (BMI from height/weight)

#### 9. No Save/Resume Workflow
- Can't save analysis and come back later
- Can't share analysis with collaborators
- No version history

#### 10. No Templates
- No saved analysis templates for common studies
- Can't reuse previous analysis on new data

---

## Phase 0 Implementation Plan

**Total Effort:** 60 hours (7.5 days)
**Can Run in Parallel with:** Phase 1 security fixes

### Milestone 0.1: Self-Service Data Upload (16 hours)

**Goal:** Doctors can upload CSV/Excel without engineering help

**Features:**
1. **Drag-and-drop file upload** (4 hours)
   - Accept CSV, Excel, SPSS (.sav)
   - Preview first 100 rows
   - Show column names and types

2. **Automatic variable type detection** (4 hours)
   - Identify: Continuous, Categorical, Binary, Date/Time
   - Let user override if wrong
   - Suggest outcome variable based on name

3. **Variable mapping wizard** (6 hours)
   - Map user's columns to standardized fields
   - "Which column is patient ID?"
   - "Which column is the outcome?"
   - "What time variables do you have?"
   - Save mapping for future use

4. **Data validation** (2 hours)
   - Check for duplicate IDs
   - Flag excessive missing data
   - Warn about potential issues

**Acceptance Criteria:**
- [ ] Doctor can upload Excel file and see data within 2 minutes
- [ ] No code or YAML editing required
- [ ] Uploaded datasets persist between sessions
- [ ] Clear error messages if data has issues

**Files to Create:**
```
src/clinical_analytics/ui/
├── pages/
│   └── 1_📤_Upload_Data.py       # NEW: Data upload page
├── components/
│   ├── file_uploader.py          # NEW: Upload widget
│   ├── variable_mapper.py        # NEW: Variable mapping UI
│   └── data_validator.py         # NEW: Data validation
└── storage/
    └── user_datasets.py          # NEW: Store uploaded data
```

**User Flow:**
```
1. Click "Upload New Data" button
2. Drag Excel file into upload area
3. System shows preview:
   ┌─────────────────────────────────┐
   │ Preview (first 5 rows):         │
   │                                 │
   │ patient_id  age  sex  outcome   │
   │ 001         45   M    1         │
   │ 002         62   F    0         │
   │ ...                             │
   └─────────────────────────────────┘
4. System asks: "What type of data?"
   ○ Patient outcomes study
   ○ Survival/time-to-event
   ○ Descriptive statistics only
5. Map columns:
   "Which column is patient ID?" → [patient_id▼]
   "Which column is outcome?"    → [outcome▼]
   "Which is treatment/exposure?" → [treatment▼]
6. Click "Start Analysis" → Data ready to use
```

---

### Milestone 0.2: Analysis Wizard (20 hours)

**Goal:** Guide doctors to appropriate statistical tests

**Features:**
1. **"I want to..." entry point** (4 hours)
   - Simple question-based navigation
   - No statistical jargon
   - Visual icons for each analysis type

2. **Smart analysis suggestions** (6 hours)
   - Based on variable types
   - "You have a binary outcome and multiple predictors → Logistic Regression"
   - "You have time-to-event data → Survival Analysis"
   - Explain why each test is appropriate

3. **All analysis types implemented** (8 hours)
   - Descriptive statistics (Table 1)
   - Group comparisons (t-test, chi-square)
   - Logistic regression (already exists)
   - Survival analysis (integrate existing code)
   - Correlation matrix
   - Subgroup analysis

4. **Progressive disclosure** (2 hours)
   - Start simple, reveal advanced options if needed
   - "Advanced Options >" expander
   - Sensible defaults for everything

**Acceptance Criteria:**
- [ ] Doctor can complete analysis in <5 clicks
- [ ] Appropriate tests suggested automatically
- [ ] Can run analysis without understanding statistics terminology
- [ ] All common clinical analyses available

**Files to Modify/Create:**
```
src/clinical_analytics/ui/
├── pages/
│   ├── 2_📊_Descriptive_Stats.py     # NEW
│   ├── 3_📈_Compare_Groups.py        # NEW
│   ├── 4_🎯_Risk_Factors.py          # MODIFY: Better wizard
│   └── 5_⏱️_Survival_Analysis.py     # NEW: Expose existing code
└── components/
    ├── analysis_wizard.py            # NEW: Guide to right test
    └── result_interpreter.py         # NEW: Plain language results
```

**User Flow - "I want to compare outcomes between groups":**
```
1. Click "I want to..." button
2. Select "Compare outcomes between groups"
3. System asks:
   "What is your outcome?"
   ○ Binary (yes/no, alive/dead)
   ○ Continuous (age, blood pressure)
   ○ Categorical (mild/moderate/severe)

4. System detects: Binary outcome
   "What are your groups?"
   [Treatment ▼] vs [Control ▼]

5. System suggests:
   ✓ Chi-square test (for binary outcome vs categorical group)
   📖 This test compares proportions between groups

6. Click "Run Analysis" → Results with interpretation
```

---

### Milestone 0.3: Publication-Ready Exports (14 hours)

**Goal:** One-click export to Word tables and PowerPoint slides

**Features:**
1. **Formatted statistical tables** (6 hours)
   - Table 1: Demographics/descriptives
   - Table 2: Regression results with ORs, CIs, p-values
   - APA format, journal-ready
   - Export to Word (.docx) and Excel

2. **Publication-quality figures** (6 hours)
   - High-resolution PNG/PDF (300 DPI)
   - Professional styling
   - Proper axis labels and legends
   - Caption suggestions

3. **Methods section generator** (2 hours)
   - Auto-generate methods text based on analyses run
   - "Logistic regression was performed with..."
   - Include statistical software citation

**Acceptance Criteria:**
- [ ] Exported Word table can be pasted directly into manuscript
- [ ] Figures are publication-quality (300 DPI, proper sizing)
- [ ] Methods text generated automatically
- [ ] Export includes all necessary information (N, CIs, p-values)

**Files to Create:**
```
src/clinical_analytics/export/
├── __init__.py
├── table_formatter.py      # NEW: Format statistical tables
├── figure_exporter.py      # NEW: High-res figures
├── word_exporter.py        # NEW: Export to Word
└── methods_generator.py    # NEW: Generate methods text
```

**Example Exports:**

**Table Export (Word):**
```
Table 1. Patient Characteristics (N=247)

Characteristic              Overall
────────────────────────────────────────
Age, years (mean ± SD)      52.3 ± 14.2
Sex, n (%)
  Male                      132 (53.4)
  Female                    115 (46.6)
Outcome, n (%)
  Yes                       89 (36.0)
  No                        158 (64.0)
────────────────────────────────────────
```

**Figure Export (PNG 300 DPI):**
- Kaplan-Meier curve with confidence bands
- Properly labeled axes
- Legend with group sizes
- Log-rank test p-value
- Publication-ready quality

---

### Milestone 0.4: Clinical Terminology & Help (10 hours)

**Goal:** Make interface approachable for non-technical users

**Features:**
1. **Terminology translation layer** (4 hours)
   - Replace all engineering terms with clinical equivalents
   - Add tooltips with explanations
   - Glossary page

2. **Context-sensitive help** (4 hours)
   - "?" icons next to every input
   - Popup explanations
   - Example use cases

3. **Video tutorials** (2 hours)
   - 2-minute video: "Upload your first dataset"
   - 2-minute video: "Run your first analysis"
   - 2-minute video: "Export results for publication"

**Acceptance Criteria:**
- [ ] No technical jargon visible to users
- [ ] Help available within 1 click everywhere
- [ ] Doctor can learn platform in <30 minutes

**Files to Create:**
```
src/clinical_analytics/ui/
├── pages/
│   └── 📚_Help_Center.py           # NEW: Help & tutorials
├── components/
│   ├── tooltip.py                  # NEW: Contextual help
│   └── glossary.py                 # NEW: Term definitions
└── assets/
    └── tutorials/                  # NEW: Video tutorials
        ├── upload_data.mp4
        ├── first_analysis.mp4
        └── export_results.mp4
```

---

## Revised User Journey (After Phase 0)

### Current Journey (Engineer-Focused):
```
❌ Too Technical, Blocks Clinical Users

1. Email engineer to add dataset
2. Wait days for engineer
3. Open complex UI with technical terms
4. Struggle to find appropriate analysis
5. Run logistic regression (only option)
6. Download raw CSV
7. Manually format results in Excel (hours)
8. Create figures in GraphPad/R (more hours)
9. Write methods section manually
```

### Target Journey (Clinician-Focused):
```
✅ Self-Service, Intuitive, Fast

1. Upload Excel file (2 minutes)
2. Click "I want to compare survival between groups"
3. System guides to Kaplan-Meier analysis
4. Click "Run Analysis" → See results with plain-English interpretation
5. Click "Export for Publication" → Get:
   - Word table (journal-ready)
   - High-res figure (300 DPI)
   - Methods text (ready to paste)
6. Paste into manuscript
7. Submit paper! 🎉

Time saved: 8+ hours → 30 minutes
```

---

## Implementation Priority Order

### Week 0 (Critical Path):
```
Priority 1: Data Upload (Milestone 0.1)
└─ Blocks everything else

Priority 2: Analysis Wizard (Milestone 0.2)
└─ Core functionality

Priority 3: Export Formatting (Milestone 0.3)
└─ Differentiator, high value

Priority 4: Help & Terminology (Milestone 0.4)
└─ Polish, can be ongoing
```

### Can Be Parallel with Phase 1:
- Upload feature (independent)
- Export formatting (independent)
- Terminology updates (low risk)

### Must Complete Before Phase 1 Ends:
- Analysis wizard (users need this to use the tool)

---

## Success Metrics (Phase 0)

### Usability Metrics:
- **Time to First Analysis:** <10 minutes (currently: days)
- **Analyses per Session:** >2 (currently: ~1)
- **Export Usage:** >80% of analyses exported
- **Support Requests:** <1 per 10 users (currently: continuous hand-holding)

### User Testing Criteria:
```
Test: Give doctor with no programming experience the tool

Pass Criteria:
- [ ] Can upload own data without asking for help
- [ ] Can run appropriate analysis within 15 minutes
- [ ] Can export publication-ready results
- [ ] Says "I would use this for my research"
```

### Clinical Validation:
- [ ] Doctor successfully creates Table 1 for manuscript
- [ ] Doctor successfully runs regression analysis
- [ ] Doctor successfully exports figures for poster
- [ ] Results match what they would get from SPSS/Stata/R

---

## Dependencies & Integration

### Phase 0 → Phase 1 Integration:
```
Phase 0 adds features, Phase 1 secures them:

Upload Feature (0.1) ──→ Authentication (1.003) ──→ Audit Log (2.009)
                    └──→ Path Validation (1.005)
                    └──→ Input Validation (3.011)

Analysis Wizard (0.2) ──→ Statistical Tests (1.006)
                      └──→ Error Handling (2.010)

Export Feature (0.3) ──→ Audit Logging (2.009)
                     └──→ Permission Checks (1.003)
```

### Key Point:
**Phase 0 can run in parallel with Phase 1**, but must complete before public release. Security (Phase 1) is foundation, usability (Phase 0) is the interface.

---

## Risk Management

### Risk 1: Upload Feature Introduces Security Issues
**Mitigation:**
- Implement with Phase 1 path validation from the start
- File size limits (100MB max)
- Only accept known formats (CSV, Excel, SPSS)
- Scan uploads for malicious content

**Rollback Plan:** Deploy behind authentication first

### Risk 2: Analysis Wizard Suggests Wrong Test
**Mitigation:**
- Always show assumptions and limitations
- Let user override suggestions
- Include disclaimer about statistical consultation
- Validate with clinical statistician

**Rollback Plan:** Default to current manual selection

### Risk 3: Export Formatting Incorrect
**Mitigation:**
- Validate exports against published papers
- Statistical review of all formatting
- Include raw data alongside formatted output

**Rollback Plan:** Keep raw CSV export available

---

## Resource Requirements

### Skill Mix:
- **Frontend Developer:** 30 hours (UI/UX, Streamlit)
- **Backend Developer:** 20 hours (Upload, storage, integration)
- **Clinical Informaticist:** 10 hours (Terminology, workflow validation)

### External Reviews Needed:
- **Clinician User Testing:** 5 researchers, 1 hour each
- **Statistician Review:** Validate analysis suggestions and output formats
- **Medical Writer Review:** Validate table/methods formatting

---

## Documentation Updates

### New Documentation Needed:
1. **User Guide for Clinicians** (replace technical docs)
   - "How to upload your data"
   - "Choosing the right analysis"
   - "Interpreting your results"
   - "Exporting for publication"

2. **Video Tutorials**
   - Screen recordings with voiceover
   - <3 minutes each
   - Focus on common tasks

3. **FAQ Section**
   - "What file formats are supported?"
   - "How do I know which test to use?"
   - "Can I export to my preferred journal format?"
   - "Is my data secure?"

4. **Statistical Methods Reference**
   - When to use each test
   - Assumptions and limitations
   - How to report results

---

## Phase 0 Deliverables Checklist

### Must Have (Blocks Release):
- [ ] Self-service data upload (no code required)
- [ ] All major analysis types available (descriptive, comparison, regression, survival)
- [ ] Analysis wizard that guides users
- [ ] Publication-ready table exports (Word format)
- [ ] Clinical terminology throughout (no jargon)
- [ ] User testing with 3+ clinicians (all successful)

### Should Have (High Value):
- [ ] High-resolution figure exports (300 DPI)
- [ ] Methods section generator
- [ ] Plain-language result interpretation
- [ ] Context-sensitive help throughout

### Nice to Have (Can Add Later):
- [ ] Video tutorials
- [ ] Save/resume analysis workflows
- [ ] Analysis templates
- [ ] Collaboration features

---

## Go/No-Go Decision Point

**Before Phase 1 Deployment:**

✅ **GO if:**
- Doctor can upload data and run analysis independently
- Exports are publication-ready
- User testing shows >80% success rate
- All "Must Have" criteria met

❌ **NO-GO if:**
- Still requires engineering support for datasets
- Exports require manual formatting
- User testing shows confusion/frustration
- Analysis suggestions frequently wrong

**Decision Maker:** Product Owner + Clinical Stakeholder

---

## Post-Phase 0 Roadmap

### Phase 0.5 (Enhancement Ideas):
- **Templates:** "Cohort Study Template", "Case-Control Template"
- **Collaboration:** Share analyses with co-authors
- **Power Calculations:** Sample size planning
- **Data Cleaning:** GUI for handling missing data, outliers
- **Multi-variable Selection:** Forward/backward selection wizards
- **Meta-Analysis Tools:** Combine results from multiple studies

### Long-Term Vision:
**"The Excel of Clinical Research"** - As easy to use as Excel, but statistically rigorous and publication-ready.

---

**END OF PHASE 0 PLAN**

**Next Steps:**
1. Review this plan with clinical stakeholders
2. Prioritize features with user input
3. Start with Milestone 0.1 (Upload)
4. Run parallel with Phase 1 security work
5. User test after each milestone
