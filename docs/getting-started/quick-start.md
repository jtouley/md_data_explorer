# Quick Start

## Your First Analysis

### Step 1: Launch the Application

```bash
streamlit run src/clinical_analytics/ui/app.py
```

### Step 2: Upload Your Data

1. Click "Upload Dataset" in the sidebar
2. Select a CSV or Excel file with clinical data
3. The platform automatically detects:
   - Patient ID column
   - Outcome variables (binary, e.g., mortality, readmission)
   - Time columns (dates, survival time)
   - Categorical and continuous variables

### Step 3: Ask a Question

Instead of navigating through menus, just ask what you want to know:

**Example Questions:**

- "Compare survival by treatment arm"
- "What predicts mortality?"
- "Show correlation between age and outcome"
- "Descriptive statistics"

The platform will:

1. Parse your question
2. Identify the appropriate analysis type
3. Extract relevant variables
4. Run the statistical test
5. Display results with plain language interpretation

### Step 4: Explore Results

Results include:

- **Statistical Test Results**: p-values, effect sizes, confidence intervals
- **Visualizations**: Plots tailored to the analysis type
- **Plain Language Summary**: What the results mean clinically
- **Export Options**: Download results as CSV or images

## Example Datasets

Try these built-in datasets to get started:

### COVID-MS Dataset

Patient-level data on COVID-19 outcomes in multiple sclerosis patients.

**Sample Questions:**

- "Compare hospitalization by treatment status"
- "What predicts ICU admission?"

### Sepsis Challenge Dataset

PhysioNet sepsis prediction challenge data.

**Sample Questions:**

- "What predicts sepsis?"
- "Compare outcomes by age group"

## Next Steps

- Learn about [Question-Driven Analysis](../user-guide/question-driven-analysis.md)
- Understand [Statistical Tests](../user-guide/statistical-tests.md)
- Upload your own data following the [Data Upload Guide](uploading-data.md)
