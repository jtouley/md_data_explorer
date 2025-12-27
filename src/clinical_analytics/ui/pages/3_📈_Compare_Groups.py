"""
Group Comparison Page

Compare outcomes or characteristics between groups using appropriate statistical tests.
Automatically selects the right test based on your data types.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.core.schema import UnifiedCohort
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.ui.components.analysis_wizard import AnalysisWizard
from clinical_analytics.ui.components.result_interpreter import ResultInterpreter

# Page config
st.set_page_config(page_title="Compare Groups | Clinical Analytics", page_icon="📈", layout="wide")


def perform_comparison(df: pd.DataFrame, outcome_col: str, group_col: str) -> Dict[str, any]:
    """
    Perform appropriate statistical comparison based on data types.

    Args:
        df: DataFrame with data
        outcome_col: Outcome variable
        group_col: Grouping variable

    Returns:
        Dictionary with test results
    """
    # Clean data
    analysis_df = df[[outcome_col, group_col]].dropna()

    if len(analysis_df) < 2:
        return {"error": "Insufficient data for analysis"}

    # Determine data types
    outcome_numeric = pd.api.types.is_numeric_dtype(analysis_df[outcome_col])
    groups = analysis_df[group_col].unique()
    n_groups = len(groups)

    if n_groups < 2:
        return {"error": "Need at least 2 groups for comparison"}

    # Select appropriate test
    if outcome_numeric:
        # Continuous outcome
        if n_groups == 2:
            # Two groups: t-test
            group1_data = analysis_df[analysis_df[group_col] == groups[0]][outcome_col]
            group2_data = analysis_df[analysis_df[group_col] == groups[1]][outcome_col]

            # Perform t-test
            statistic, p_value = stats.ttest_ind(group1_data, group2_data)

            # Calculate means and CIs
            mean1 = group1_data.mean()
            mean2 = group2_data.mean()
            std1 = group1_data.std()
            std2 = group2_data.std()
            n1 = len(group1_data)
            n2 = len(group2_data)

            # Mean difference and CI
            mean_diff = mean1 - mean2
            se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff

            return {
                "test": "Independent Samples T-Test",
                "test_type": "t-test",
                "outcome_type": "continuous",
                "statistic": statistic,
                "p_value": p_value,
                "groups": {
                    str(groups[0]): {"mean": mean1, "std": std1, "n": n1},
                    str(groups[1]): {"mean": mean2, "std": std2, "n": n2},
                },
                "mean_difference": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }

        else:
            # More than 2 groups: ANOVA
            group_data = [analysis_df[analysis_df[group_col] == g][outcome_col] for g in groups]

            # Perform ANOVA
            statistic, p_value = stats.f_oneway(*group_data)

            # Calculate group means
            group_stats = {}
            for g in groups:
                g_data = analysis_df[analysis_df[group_col] == g][outcome_col]
                group_stats[str(g)] = {"mean": g_data.mean(), "std": g_data.std(), "n": len(g_data)}

            return {
                "test": "One-Way ANOVA",
                "test_type": "anova",
                "outcome_type": "continuous",
                "statistic": statistic,
                "p_value": p_value,
                "groups": group_stats,
            }

    else:
        # Categorical outcome: Chi-square test
        # Create contingency table
        contingency = pd.crosstab(analysis_df[outcome_col], analysis_df[group_col])

        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Calculate proportions for each group
        group_props = {}
        for g in groups:
            g_data = analysis_df[analysis_df[group_col] == g][outcome_col]
            value_counts = g_data.value_counts()
            total = len(g_data)

            group_props[str(g)] = {
                "counts": value_counts.to_dict(),
                "proportions": (value_counts / total).to_dict(),
                "n": total,
            }

        return {
            "test": "Chi-Square Test",
            "test_type": "chi-square",
            "outcome_type": "categorical",
            "statistic": chi2,
            "p_value": p_value,
            "dof": dof,
            "groups": group_props,
            "contingency_table": contingency,
        }


def main():
    st.title("📈 Compare Groups")
    st.markdown("""
    Compare outcomes or characteristics between groups.
    We'll automatically select the right statistical test for your data.
    """)

    # Dataset selection
    st.sidebar.header("Data Selection")

    # Load datasets (same as descriptive stats)
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
    except:
        pass

    if not dataset_display_names:
        st.error("No datasets available. Please upload data first.")
        return

    dataset_choice_display = st.sidebar.selectbox(
        "Choose Dataset", list(dataset_display_names.keys())
    )
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

    # Variable selection
    st.markdown("## 🔧 Configure Comparison")

    available_cols = [
        c for c in cohort.columns if c not in [UnifiedCohort.PATIENT_ID, UnifiedCohort.TIME_ZERO]
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What do you want to compare?")
        outcome_col = st.selectbox(
            "Outcome Variable",
            available_cols,
            help="The variable you want to compare between groups",
        )

    with col2:
        st.markdown("### Between which groups?")
        group_col = st.selectbox(
            "Grouping Variable",
            [c for c in available_cols if c != outcome_col],
            help="The variable that defines your groups (e.g., treatment, sex, age group)",
        )

    if not outcome_col or not group_col:
        st.warning("Please select both outcome and grouping variables")
        return

    # Show data preview
    with st.expander("👀 Preview Data"):
        preview_df = cohort[[outcome_col, group_col]].dropna()
        st.dataframe(preview_df.head(20))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", len(preview_df))
        with col2:
            st.metric("Groups", preview_df[group_col].nunique())
        with col3:
            st.metric("Complete Cases", len(preview_df))

    # Run analysis
    if st.button("📊 Compare Groups", type="primary"):
        with st.spinner("Running statistical test..."):
            results = perform_comparison(cohort, outcome_col, group_col)

            if "error" in results:
                st.error(results["error"])
                return

            st.success(f"✅ Analysis complete: {results['test']}")

            # Explain test choice
            AnalysisWizard.explain_test_choice(
                test_name=results["test"],
                variable_types={outcome_col: results["outcome_type"], group_col: "grouping"},
                outcome_type=results["outcome_type"],
            )

            # Display results
            st.markdown("## 📊 Results")

            if results["test_type"] == "t-test":
                # T-test results
                groups_list = list(results["groups"].keys())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("T-statistic", f"{results['statistic']:.3f}")
                with col2:
                    st.metric("P-value", f"{results['p_value']:.4f}")
                with col3:
                    p_interp = ResultInterpreter.interpret_p_value(results["p_value"])
                    st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

                # Group statistics
                st.markdown("### Group Statistics")

                col1, col2 = st.columns(2)
                for idx, (group, stats) in enumerate(results["groups"].items()):
                    with [col1, col2][idx]:
                        st.markdown(f"**{group}**")
                        st.markdown(f"- Mean: {stats['mean']:.2f}")
                        st.markdown(f"- SD: {stats['std']:.2f}")
                        st.markdown(f"- N: {stats['n']}")

                # Interpretation
                st.markdown("### 📖 Interpretation")
                interpretation = ResultInterpreter.interpret_mean_difference(
                    mean_diff=results["mean_difference"],
                    ci_lower=results["ci_lower"],
                    ci_upper=results["ci_upper"],
                    p_value=results["p_value"],
                    group1=groups_list[0],
                    group2=groups_list[1],
                    outcome_name=outcome_col,
                )
                st.markdown(interpretation)

            elif results["test_type"] == "anova":
                # ANOVA results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("F-statistic", f"{results['statistic']:.3f}")
                with col2:
                    st.metric("P-value", f"{results['p_value']:.4f}")
                with col3:
                    p_interp = ResultInterpreter.interpret_p_value(results["p_value"])
                    st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

                # Group statistics
                st.markdown("### Group Statistics")
                group_df = pd.DataFrame(results["groups"]).T
                st.dataframe(group_df)

                # Interpretation
                st.markdown("### 📖 Interpretation")
                p_interp = ResultInterpreter.interpret_p_value(results["p_value"])

                if p_interp["is_significant"]:
                    st.markdown(f"""
**Significant difference found** {p_interp["emoji"]}

The ANOVA test shows that at least one group differs significantly from the others (p={results["p_value"]:.4f}).

**Next steps**: Perform post-hoc tests (e.g., Tukey's HSD) to identify which specific groups differ from each other.
""")
                else:
                    st.markdown(f"""
**No significant difference** ❌

The ANOVA test shows no significant difference in {outcome_col} across groups (p={results["p_value"]:.4f}).
All groups appear similar on this measure.
""")

            elif results["test_type"] == "chi-square":
                # Chi-square results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chi-square", f"{results['statistic']:.3f}")
                with col2:
                    st.metric("P-value", f"{results['p_value']:.4f}")
                with col3:
                    p_interp = ResultInterpreter.interpret_p_value(results["p_value"])
                    st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

                # Contingency table
                st.markdown("### Contingency Table")
                st.dataframe(results["contingency_table"])

                # Group proportions
                st.markdown("### Group Proportions")
                for group, stats in results["groups"].items():
                    st.markdown(f"**{group}** (N={stats['n']})")
                    for value, prop in stats["proportions"].items():
                        count = stats["counts"][value]
                        st.markdown(f"- {value}: {count} ({prop * 100:.1f}%)")

                # Interpretation
                st.markdown("### 📖 Interpretation")
                p_interp = ResultInterpreter.interpret_p_value(results["p_value"])

                if p_interp["is_significant"]:
                    st.markdown(f"""
**Significant association found** {p_interp["emoji"]}

The chi-square test shows a significant association between {group_col} and {outcome_col} (χ²={results["statistic"]:.2f}, p={results["p_value"]:.4f}).

The distribution of {outcome_col} differs significantly across {group_col} groups.
""")
                else:
                    st.markdown(f"""
**No significant association** ❌

The chi-square test shows no significant association between {group_col} and {outcome_col} (χ²={results["statistic"]:.2f}, p={results["p_value"]:.4f}).

The distribution of {outcome_col} is similar across groups.
""")

            # Export and methods
            st.markdown("## 📥 Export & Methods")

            col1, col2 = st.columns(2)

            with col1:
                # Export results
                results_df = pd.DataFrame(
                    {
                        "Test": [results["test"]],
                        "Statistic": [results["statistic"]],
                        "P-value": [results["p_value"]],
                    }
                )

                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv_data,
                    f"comparison_{outcome_col}_by_{group_col}.csv",
                    "text/csv",
                )

            with col2:
                # Methods text
                methods_text = ResultInterpreter.generate_methods_text(
                    analysis_type="group_comparison",
                    test_name=results["test"],
                    variables={"outcome": outcome_col, "groups": group_col},
                )

                st.download_button(
                    "Download Methods Text", methods_text, "methods_comparison.txt", "text/plain"
                )


if __name__ == "__main__":
    main()
