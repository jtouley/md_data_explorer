"""
Risk Factors Analysis Page

Identify which variables predict an outcome using regression analysis.
Automatically selects logistic or linear regression based on outcome type.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Keep only lightweight imports at module scope
from clinical_analytics.core.schema import UnifiedCohort

# Heavy imports moved inside main() after gate

# Page config
st.set_page_config(page_title="Risk Factors | Clinical Analytics", page_icon="🎯", layout="wide")


def main():
    # Gate: V1 MVP mode disables legacy pages
    # MUST run before any expensive operations
    from clinical_analytics.ui.config import V1_MVP_MODE

    if V1_MVP_MODE:
        st.info("🚧 This page is disabled in V1 MVP mode. Use the **Ask Questions** page for all analysis.")
        st.markdown("""
        **V1 MVP focuses on:**
        - Upload your data
        - Ask questions in natural language
        - Get answers with SQL preview

        All analysis is available through the Chat interface on the Ask Questions page.
        """)
        if st.button("Go to Ask Questions Page"):
            st.switch_page("pages/3_💬_Ask_Questions.py")
        st.stop()

    # NOW do heavy imports (after gate)
    from clinical_analytics.analysis.stats import run_logistic_regression
    from clinical_analytics.core.registry import DatasetRegistry
    from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
    from clinical_analytics.ui.components.analysis_wizard import AnalysisWizard
    from clinical_analytics.ui.components.result_interpreter import ResultInterpreter
    from clinical_analytics.ui.helpers import require_outcome

    st.title("🎯 Identify Risk Factors")
    st.markdown("""
    Find which variables predict an outcome. We'll use **regression analysis** to identify
    risk factors and calculate odds ratios or effect sizes.
    """)

    # Dataset selection
    st.sidebar.header("Data Selection")

    # Load datasets
    available_datasets = DatasetRegistry.list_datasets()
    dataset_info = DatasetRegistry.get_all_dataset_info()

    dataset_display_names = {}
    for ds_name in available_datasets:
        info = dataset_info[ds_name]
        display_name = info["config"].get("display_name", ds_name.replace("_", "-").upper())
        dataset_display_names[display_name] = ds_name

    uploaded_datasets = {}
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            upload_id = upload["upload_id"]
            dataset_name = upload.get("dataset_name", upload_id)
            display_name = f"📤 {dataset_name}"
            dataset_display_names[display_name] = upload_id
            uploaded_datasets[upload_id] = upload
    except Exception:
        pass

    if not dataset_display_names:
        st.error("No datasets available. Please upload data first.")
        return

    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))
    dataset_choice = dataset_display_names[dataset_choice_display]
    is_uploaded = dataset_choice in uploaded_datasets

    # Load dataset
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            if is_uploaded:
                dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
                dataset.load()
            else:
                dataset = DatasetRegistry.get_dataset(dataset_choice)
                dataset.validate()
                dataset.load()

            cohort = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    # Check if outcome is required and available
    require_outcome(cohort, "Risk Factors analysis")

    # Configuration
    st.markdown("## 🔧 Configure Analysis")

    available_cols = [c for c in cohort.columns if c not in [UnifiedCohort.PATIENT_ID, UnifiedCohort.TIME_ZERO]]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What outcome do you want to predict?")

        # Try to default to 'outcome' column if it exists
        default_outcome = None
        if UnifiedCohort.OUTCOME in cohort.columns:
            default_outcome = UnifiedCohort.OUTCOME
        elif "outcome" in [c.lower() for c in cohort.columns]:
            default_outcome = [c for c in cohort.columns if c.lower() == "outcome"][0]

        outcome_col = st.selectbox(
            "Outcome Variable",
            available_cols,
            index=available_cols.index(default_outcome) if default_outcome and default_outcome in available_cols else 0,
            help="The outcome you want to predict (must be binary for logistic regression)",
        )

        # Check outcome type
        if outcome_col:
            outcome_data = cohort[outcome_col].dropna()
            n_unique = outcome_data.nunique()

            if n_unique == 2:
                st.success(f"✅ Binary outcome detected ({n_unique} values) - will use **Logistic Regression**")
                regression_type = "logistic"
            elif n_unique <= 10:
                st.warning(f"⚠️ Outcome has {n_unique} values. Consider if this should be binary.")
                regression_type = "logistic"
            else:
                st.info("ℹ️ Continuous outcome detected - would use Linear Regression (not yet implemented)")
                regression_type = "linear"

    with col2:
        st.markdown("### Which variables might predict the outcome?")

        available_predictors = [c for c in available_cols if c != outcome_col]

        selected_predictors = st.multiselect(
            "Predictor Variables (Risk Factors)",
            available_predictors,
            default=available_predictors[: min(5, len(available_predictors))],
            help="Select variables that might predict or influence the outcome",
        )

        if selected_predictors:
            st.caption(f"Selected {len(selected_predictors)} predictor(s)")

    # Data preview
    with st.expander("👀 Preview Data"):
        if selected_predictors:
            preview_cols = [outcome_col] + selected_predictors
            preview_df = cohort[preview_cols].dropna()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Complete Cases", len(preview_df))
            with col2:
                if outcome_col in preview_df:
                    outcome_rate = preview_df[outcome_col].mean() * 100
                    st.metric("Outcome Rate", f"{outcome_rate:.1f}%")
            with col3:
                st.metric("Predictors", len(selected_predictors))

            st.dataframe(preview_df.head(10))

    # Run analysis
    if not selected_predictors:
        st.warning("Please select at least one predictor variable")
        return

    if st.button("🎯 Identify Risk Factors", type="primary"):
        if regression_type == "linear":
            st.error("Linear regression not yet implemented. Please select a binary outcome.")
            return

        with st.spinner("Running logistic regression..."):
            try:
                # Prepare data
                analysis_cols = [outcome_col] + selected_predictors
                analysis_df = cohort[analysis_cols].copy()

                # Handle categorical variables
                categorical_cols = analysis_df.select_dtypes(include=["object", "category"]).columns.tolist()
                if outcome_col in categorical_cols:
                    categorical_cols.remove(outcome_col)

                if categorical_cols:
                    st.info(f"Converting categorical variables to dummy variables: {', '.join(categorical_cols)}")
                    analysis_df = pd.get_dummies(analysis_df, columns=categorical_cols, drop_first=True)

                    # Update predictor list with dummy variable names
                    new_predictors = [c for c in analysis_df.columns if c != outcome_col]
                else:
                    new_predictors = selected_predictors

                # Drop missing values
                initial_n = len(analysis_df)
                analysis_df = analysis_df.dropna()
                final_n = len(analysis_df)

                if final_n < initial_n:
                    st.warning(
                        f"Dropped {initial_n - final_n} rows with missing data "
                        f"({(initial_n - final_n) / initial_n * 100:.1f}%)"
                    )

                if final_n < 10:
                    st.error("Insufficient data after cleaning. Need at least 10 complete cases.")
                    return

                # Run regression
                model, summary_df = run_logistic_regression(analysis_df, outcome_col, new_predictors)

                st.success("✅ Analysis complete!")

                # Explain test choice
                AnalysisWizard.explain_test_choice(
                    test_name="Logistic Regression",
                    variable_types={outcome_col: "binary"},
                    outcome_type="binary",
                )

                # Display results
                st.markdown("## 📊 Results")

                # Model statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pseudo R²", f"{model.prsquared:.4f}")
                with col2:
                    st.metric("Log-Likelihood", f"{model.llf:.2f}")
                with col3:
                    st.metric("Sample Size", final_n)

                # Results table
                st.markdown("### Odds Ratios & Confidence Intervals")
                st.dataframe(
                    summary_df.style.format(
                        {
                            "Odds Ratio": "{:.3f}",
                            "CI Lower": "{:.3f}",
                            "CI Upper": "{:.3f}",
                            "P-Value": "{:.4f}",
                        }
                    )
                )

                # Interpretation for each variable
                st.markdown("### 📖 Interpretation")

                for var in summary_df.index:
                    if var == "Intercept":
                        continue

                    or_val = summary_df.loc[var, "Odds Ratio"]
                    ci_lower = summary_df.loc[var, "CI Lower"]
                    ci_upper = summary_df.loc[var, "CI Upper"]
                    p_val = summary_df.loc[var, "P-Value"]

                    interpretation = ResultInterpreter.interpret_odds_ratio(
                        or_value=or_val,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        p_value=p_val,
                        variable_name=var,
                    )

                    with st.expander(f"**{var}**"):
                        st.markdown(interpretation)

                # Model summary
                with st.expander("📋 Full Model Summary"):
                    st.text(str(model.summary()))

                # Export options
                st.markdown("## 📥 Export & Methods")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # CSV export
                    csv_data = summary_df.to_csv()
                    st.download_button(
                        "Download Results CSV",
                        csv_data,
                        f"risk_factors_{dataset_choice}.csv",
                        "text/csv",
                    )

                with col2:
                    # Full summary export
                    model_summary_text = str(model.summary())
                    st.download_button(
                        "Download Full Summary",
                        model_summary_text,
                        f"model_summary_{dataset_choice}.txt",
                        "text/plain",
                    )

                with col3:
                    # Methods text
                    methods_text = ResultInterpreter.generate_methods_text(
                        analysis_type="regression",
                        test_name="Logistic Regression",
                        variables={"outcome": outcome_col},
                    )

                    st.download_button(
                        "Download Methods Text",
                        methods_text,
                        "methods_regression.txt",
                        "text/plain",
                    )

            except Exception as e:
                st.error(f"Error running analysis: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
