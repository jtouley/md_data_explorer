"""
Survival Analysis Page

Analyze time-to-event data using Kaplan-Meier curves and Cox regression.
Understand how long patients survive or how quickly events occur.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Keep only lightweight imports at module scope
from clinical_analytics.core.schema import UnifiedCohort

# Heavy imports moved inside main() after gate

# Page config
st.set_page_config(page_title="Survival Analysis | Clinical Analytics", page_icon="⏱️", layout="wide")


def plot_kaplan_meier(kmf, summary_df: pd.DataFrame, group_col: str = None):
    """
    Create Kaplan-Meier survival curve plot.

    Args:
        kmf: Fitted KaplanMeierFitter model
        summary_df: Summary DataFrame with survival probabilities
        group_col: Optional group column name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if group_col is None:
        # Single curve
        ax.plot(summary_df["time"], summary_df["survival_probability"], label="Overall", linewidth=2)
        ax.fill_between(summary_df["time"], summary_df["ci_lower"], summary_df["ci_upper"], alpha=0.2)
    else:
        # Multiple curves by group
        for group in summary_df["group"].unique():
            group_data = summary_df[summary_df["group"] == group]
            ax.plot(
                group_data["time"],
                group_data["survival_probability"],
                label=str(group),
                linewidth=2,
            )
            ax.fill_between(group_data["time"], group_data["ci_lower"], group_data["ci_upper"], alpha=0.2)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title("Kaplan-Meier Survival Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    return fig


def main():
    # Gate: V1 MVP mode disables legacy pages
    # MUST run before any expensive operations
    from clinical_analytics.ui.helpers import gate_v1_mvp_legacy_page

    gate_v1_mvp_legacy_page()  # Stops execution if gated

    # NOW do heavy imports (after gate)
    from clinical_analytics.analysis.survival import (
        calculate_median_survival,
        run_cox_regression,
        run_kaplan_meier,
        run_logrank_test,
    )
    from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
    from clinical_analytics.ui.components.analysis_wizard import AnalysisWizard
    from clinical_analytics.ui.components.result_interpreter import ResultInterpreter

    st.title("⏱️ Survival Analysis")
    st.markdown("""
    Analyze **time-to-event** data. How long do patients survive? How quickly do events occur?
    We'll use **Kaplan-Meier curves** and **Cox regression** to answer these questions.
    """)

    # Dataset selection
    st.sidebar.header("Data Selection")

    # Load datasets - only user uploads
    dataset_display_names = {}
    uploaded_datasets = {}
    try:
        uploads = UploadedDatasetFactory.list_available_uploads()
        for upload in uploads:
            upload_id = upload["upload_id"]
            dataset_name = upload.get("dataset_name", upload_id)
            display_name = f"📤 {dataset_name}"
            dataset_display_names[display_name] = upload_id
            uploaded_datasets[upload_id] = upload
    except Exception as e:
        st.sidebar.warning(f"Could not load uploaded datasets: {e}")

    if not dataset_display_names:
        st.error("No datasets found! Please upload data using the 'Add Your Data' page.")
        st.info("👈 Go to **Add Your Data** to upload your first dataset")
        return

    dataset_choice_display = st.sidebar.selectbox("Choose Dataset", list(dataset_display_names.keys()))
    dataset_choice = dataset_display_names[dataset_choice_display]

    # Load dataset (always uploaded)
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
            dataset.load()

            cohort = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    # Configuration
    st.markdown("## 🔧 Configure Analysis")

    available_cols = [c for c in cohort.columns if c not in [UnifiedCohort.PATIENT_ID, UnifiedCohort.TIME_ZERO]]

    # Analysis type selection
    analysis_type = st.radio(
        "What do you want to analyze?",
        [
            "📈 Compare survival between groups (Kaplan-Meier + Log-rank test)",
            "🎯 Find predictors of survival (Cox Regression)",
        ],
        help="Choose your analysis approach",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Time Variable")
        duration_col = st.selectbox(
            "How long until event or censoring?",
            available_cols,
            help=("Time from start until event occurs or patient is censored (e.g., days, months, years)"),
        )

    with col2:
        st.markdown("### Event Variable")
        event_col = st.selectbox(
            "Did the event occur?",
            [c for c in available_cols if c != duration_col],
            help="Binary variable: 1 = event occurred, 0 = censored (event didn't occur yet)",
        )

    # Check event variable
    if event_col:
        event_data = cohort[event_col].dropna()
        n_unique = event_data.nunique()

        if n_unique == 2:
            unique_vals = sorted(event_data.unique())
            st.success(f"✅ Binary event detected: {unique_vals[0]} and {unique_vals[1]}")

            # Ask user which value represents event
            st.info("ℹ️ Which value means the event **occurred**?")
            event_occurred_value = st.radio("Event occurred when value is:", unique_vals, horizontal=True)
        else:
            st.error(f"❌ Event variable must be binary (has {n_unique} unique values)")
            return

    if "📈 Compare survival" in analysis_type:
        # Kaplan-Meier + Log-rank
        st.markdown("### Grouping Variable (Optional)")
        group_col = st.selectbox(
            "Compare survival between which groups?",
            ["None"] + [c for c in available_cols if c not in [duration_col, event_col]],
            help="Optional: split survival curves by groups (e.g., treatment vs control)",
        )

        if group_col == "None":
            group_col = None

    else:
        # Cox regression
        st.markdown("### Predictor Variables")
        available_predictors = [c for c in available_cols if c not in [duration_col, event_col]]

        covariates = st.multiselect(
            "Which variables might predict survival time?",
            available_predictors,
            default=available_predictors[: min(5, len(available_predictors))],
            help="Select variables that might influence time to event",
        )

    # Data preview
    with st.expander("👀 Preview Data"):
        if "📈 Compare survival" in analysis_type:
            preview_cols = [duration_col, event_col]
            if group_col:
                preview_cols.append(group_col)
        else:
            preview_cols = [duration_col, event_col] + covariates[:5]  # Show first 5 predictors

        preview_df = cohort[preview_cols].dropna()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Complete Cases", len(preview_df))
        with col2:
            # Recode event variable
            event_series = preview_df[event_col]
            if event_col in preview_df.columns:
                event_binary = (event_series == event_occurred_value).astype(int)
                n_events = event_binary.sum()
                st.metric("Events", n_events)
        with col3:
            st.metric("Censored", len(preview_df) - n_events)

        st.dataframe(preview_df.head(10))

    # Run analysis
    if "📈 Compare survival" in analysis_type:
        button_text = "📈 Analyze Survival Curves"
    else:
        button_text = "🎯 Find Survival Predictors"

    if st.button(button_text, type="primary"):
        with st.spinner("Running survival analysis..."):
            try:
                # Prepare data - recode event variable
                analysis_cols = [duration_col, event_col]
                if "📈 Compare survival" in analysis_type and group_col:
                    analysis_cols.append(group_col)
                elif "🎯 Find predictors" in analysis_type:
                    analysis_cols.extend(covariates)

                analysis_df = cohort[analysis_cols].copy()

                # Recode event variable to binary 0/1
                analysis_df[event_col] = (analysis_df[event_col] == event_occurred_value).astype(int)

                # Drop missing
                initial_n = len(analysis_df)
                analysis_df = analysis_df.dropna()
                final_n = len(analysis_df)

                if final_n < initial_n:
                    st.warning(
                        f"Dropped {initial_n - final_n} rows with missing data "
                        f"({(initial_n - final_n) / initial_n * 100:.1f}%)"
                    )

                if final_n < 10:
                    st.error("Insufficient data. Need at least 10 complete cases.")
                    return

                if "📈 Compare survival" in analysis_type:
                    # Kaplan-Meier analysis
                    kmf, summary_df = run_kaplan_meier(
                        analysis_df,
                        duration_col=duration_col,
                        event_col=event_col,
                        group_col=group_col,
                    )

                    st.success("✅ Analysis complete!")

                    # Explain method
                    AnalysisWizard.explain_test_choice(
                        test_name="Kaplan-Meier" if not group_col else "Kaplan-Meier + Log-rank Test",
                        variable_types={duration_col: "time", event_col: "binary"},
                        outcome_type="time-to-event",
                    )

                    # Display results
                    st.markdown("## 📊 Results")

                    # Plot survival curves
                    st.markdown("### Survival Curves")
                    fig = plot_kaplan_meier(kmf, summary_df, group_col)
                    st.pyplot(fig)

                    # Median survival times
                    st.markdown("### Median Survival Times")
                    median_df = calculate_median_survival(
                        analysis_df,
                        duration_col=duration_col,
                        event_col=event_col,
                        group_col=group_col,
                    )
                    st.dataframe(median_df)

                    # Log-rank test if groups
                    if group_col:
                        st.markdown("### Statistical Comparison")
                        logrank_results = run_logrank_test(
                            analysis_df,
                            duration_col=duration_col,
                            event_col=event_col,
                            group_col=group_col,
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Statistic", f"{logrank_results['test_statistic']:.3f}")
                        with col2:
                            st.metric("P-value", f"{logrank_results['p_value']:.4f}")
                        with col3:
                            p_interp = ResultInterpreter.interpret_p_value(logrank_results["p_value"])
                            st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

                        # Interpretation
                        st.markdown("### 📖 Interpretation")
                        if p_interp["is_significant"]:
                            st.markdown(f"""
**Significant difference in survival** {p_interp["emoji"]}

The log-rank test shows that survival curves differ significantly between groups
(p={logrank_results["p_value"]:.4f}).

**Clinical Interpretation**: The {group_col} groups have different survival patterns.
Look at the survival curves and median survival times to see which group has better survival.
""")
                        else:
                            st.markdown(f"""
**No significant difference in survival** ❌

The log-rank test shows no significant difference in survival between groups
(p={logrank_results["p_value"]:.4f}).

**Clinical Interpretation**: The {group_col} groups have similar survival patterns.
""")

                    # Export options
                    st.markdown("## 📥 Export")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Save plot
                        from io import BytesIO

                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        buf.seek(0)

                        st.download_button(
                            "Download Survival Curve",
                            buf,
                            f"survival_curve_{dataset_choice}.png",
                            "image/png",
                        )

                    with col2:
                        # Export survival data
                        csv_data = summary_df.to_csv(index=False)
                        st.download_button(
                            "Download Survival Data",
                            csv_data,
                            f"survival_data_{dataset_choice}.csv",
                            "text/csv",
                        )

                    with col3:
                        # Methods text
                        if group_col:
                            methods = ResultInterpreter.generate_methods_text(
                                analysis_type="survival",
                                test_name="Kaplan-Meier survival analysis with log-rank test",
                                variables={"event": event_col, "groups": group_col},
                            )
                        else:
                            methods = f"""
Kaplan-Meier survival analysis was performed to estimate survival probabilities over time.
Time to event was measured using {duration_col}, with {event_col} as the event indicator.
Median survival times with 95% confidence intervals were calculated. All analyses were
performed using Clinical Analytics Platform.
"""

                        st.download_button("Download Methods Text", methods, "methods_survival.txt", "text/plain")

                else:
                    # Cox regression
                    if not covariates:
                        st.warning("Please select at least one predictor variable")
                        return

                    # Handle categorical variables
                    categorical_cols = analysis_df.select_dtypes(include=["object", "category"]).columns.tolist()
                    categorical_cols = [c for c in categorical_cols if c in covariates]

                    if categorical_cols:
                        st.info(f"Converting categorical variables to dummy variables: {', '.join(categorical_cols)}")

                    cph, summary_df = run_cox_regression(
                        analysis_df,
                        duration_col=duration_col,
                        event_col=event_col,
                        covariates=covariates,
                    )

                    st.success("✅ Analysis complete!")

                    # Explain method
                    AnalysisWizard.explain_test_choice(
                        test_name="Cox Regression",
                        variable_types={event_col: "time-to-event"},
                        outcome_type="survival",
                    )

                    # Display results
                    st.markdown("## 📊 Results")

                    # Model statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Concordance", f"{cph.concordance_index_:.4f}")
                    with col2:
                        st.metric("Log-Likelihood", f"{cph.log_likelihood_:.2f}")
                    with col3:
                        st.metric("Sample Size", final_n)

                    # Results table
                    st.markdown("### Hazard Ratios & Confidence Intervals")
                    display_df = summary_df.copy()
                    display_df.columns = [
                        "Hazard Ratio",
                        "HR CI Lower",
                        "HR CI Upper",
                        "P-Value",
                        "Coefficient",
                        "SE",
                        "Z-score",
                    ]
                    st.dataframe(
                        display_df.style.format(
                            {
                                "Hazard Ratio": "{:.3f}",
                                "HR CI Lower": "{:.3f}",
                                "HR CI Upper": "{:.3f}",
                                "P-Value": "{:.4f}",
                                "Coefficient": "{:.4f}",
                                "SE": "{:.4f}",
                                "Z-score": "{:.3f}",
                            }
                        )
                    )

                    # Interpretation for each variable
                    st.markdown("### 📖 Interpretation")

                    for var in summary_df.index:
                        hr_val = summary_df.loc[var, "hazard_ratio"]
                        ci_lower = summary_df.loc[var, "hr_ci_lower"]
                        ci_upper = summary_df.loc[var, "hr_ci_upper"]
                        p_val = summary_df.loc[var, "p"]

                        interpretation = ResultInterpreter.interpret_hazard_ratio(
                            hr_value=hr_val,
                            ci_lower=ci_lower,
                            ci_upper=ci_upper,
                            p_value=p_val,
                            variable_name=var,
                        )

                        with st.expander(f"**{var}**"):
                            st.markdown(interpretation)

                    # Model summary
                    with st.expander("📋 Full Model Summary"):
                        st.text(str(cph.summary))

                    # Export options
                    st.markdown("## 📥 Export")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # CSV export
                        csv_data = display_df.to_csv()
                        st.download_button(
                            "Download Results CSV",
                            csv_data,
                            f"cox_regression_{dataset_choice}.csv",
                            "text/csv",
                        )

                    with col2:
                        # Full summary
                        summary_text = str(cph.summary)
                        st.download_button(
                            "Download Full Summary",
                            summary_text,
                            f"cox_summary_{dataset_choice}.txt",
                            "text/plain",
                        )

                    with col3:
                        # Methods text
                        methods_text = ResultInterpreter.generate_methods_text(
                            analysis_type="survival",
                            test_name="Cox proportional hazards regression",
                            variables={"event": event_col},
                        )

                        st.download_button("Download Methods Text", methods_text, "methods_cox.txt", "text/plain")

            except Exception as e:
                st.error(f"Error running analysis: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
