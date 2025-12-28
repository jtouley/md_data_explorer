"""
Dynamic Analysis Page - Question-Driven Analytics

Ask questions, get answers. No statistical jargon - just tell me what you want to know.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scipy import stats

# Import analysis functions
from clinical_analytics.analysis.stats import run_logistic_regression
from clinical_analytics.analysis.survival import (
    run_kaplan_meier,
)
from clinical_analytics.core.registry import DatasetRegistry
from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
from clinical_analytics.ui.components.question_engine import (
    AnalysisContext,
    AnalysisIntent,
    QuestionEngine,
)
from clinical_analytics.ui.components.result_interpreter import ResultInterpreter
from clinical_analytics.ui.config import MULTI_TABLE_ENABLED

# Page config
st.set_page_config(page_title="Ask Questions | Clinical Analytics", page_icon="💬", layout="wide")


def run_descriptive_analysis(df: pd.DataFrame, context: AnalysisContext):
    """Generate descriptive statistics."""
    st.markdown("## 📊 Your Data at a Glance")

    # Overall metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Variables", len(df.columns))
    with col3:
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        st.metric("Data Completeness", f"{100 - missing_pct:.1f}%")

    # Summary statistics
    st.markdown("### Summary Statistics")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        st.markdown("**Numeric Variables:**")
        desc_stats = df[numeric_cols].describe().T
        st.dataframe(desc_stats)

    if categorical_cols:
        st.markdown("**Categorical Variables:**")
        for col in categorical_cols[:10]:  # Limit to first 10
            value_counts = df[col].value_counts()
            st.markdown(f"**{col}:**")
            for value, count in value_counts.head(5).items():
                pct = count / len(df) * 100
                st.write(f"  - {value}: {count} ({pct:.1f}%)")


def run_comparison_analysis(df: pd.DataFrame, context: AnalysisContext):
    """Run group comparison based on variable types."""
    st.markdown("## 📈 Group Comparison")

    outcome_col = context.primary_variable
    group_col = context.grouping_variable

    # Clean data
    analysis_df = df[[outcome_col, group_col]].dropna()

    if len(analysis_df) < 2:
        st.error("Not enough data for comparison")
        return

    # Determine appropriate test
    outcome_numeric = pd.api.types.is_numeric_dtype(analysis_df[outcome_col])
    groups = analysis_df[group_col].unique()
    n_groups = len(groups)

    if n_groups < 2:
        st.error("Need at least 2 groups for comparison")
        return

    # Run appropriate test
    if outcome_numeric:
        if n_groups == 2:
            # T-test
            group1_data = analysis_df[analysis_df[group_col] == groups[0]][outcome_col]
            group2_data = analysis_df[analysis_df[group_col] == groups[1]][outcome_col]

            statistic, p_value = stats.ttest_ind(group1_data, group2_data)

            st.markdown("### Results")
            st.markdown(f"**Comparing {outcome_col} between {groups[0]} and {groups[1]}**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{groups[0]} Average", f"{group1_data.mean():.2f}")
            with col2:
                st.metric(f"{groups[1]} Average", f"{group2_data.mean():.2f}")
            with col3:
                p_interp = ResultInterpreter.interpret_p_value(p_value)
                st.metric("Difference", f"{p_interp['significance']} {p_interp['emoji']}")

            st.markdown("### What does this mean?")
            mean_diff = group1_data.mean() - group2_data.mean()
            interpretation = ResultInterpreter.interpret_mean_difference(
                mean_diff=mean_diff,
                ci_lower=mean_diff
                - 1.96
                * np.sqrt((group1_data.std() ** 2 / len(group1_data)) + (group2_data.std() ** 2 / len(group2_data))),
                ci_upper=mean_diff
                + 1.96
                * np.sqrt((group1_data.std() ** 2 / len(group1_data)) + (group2_data.std() ** 2 / len(group2_data))),
                p_value=p_value,
                group1=str(groups[0]),
                group2=str(groups[1]),
                outcome_name=outcome_col,
            )
            st.markdown(interpretation)

        else:
            # ANOVA
            group_data = [analysis_df[analysis_df[group_col] == g][outcome_col] for g in groups]
            statistic, p_value = stats.f_oneway(*group_data)

            st.markdown("### Results")
            st.markdown(f"**Comparing {outcome_col} across {n_groups} groups**")

            p_interp = ResultInterpreter.interpret_p_value(p_value)
            st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

            if p_interp["is_significant"]:
                st.success("✅ The groups differ significantly")
                st.markdown("**Group Averages:**")
                for g in groups:
                    g_mean = analysis_df[analysis_df[group_col] == g][outcome_col].mean()
                    st.write(f"- {g}: {g_mean:.2f}")
            else:
                st.info("ℹ️ No significant difference between groups")

    else:
        # Chi-square test for categorical outcome
        contingency = pd.crosstab(analysis_df[outcome_col], analysis_df[group_col])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        st.markdown("### Results")
        st.markdown(f"**Association between {outcome_col} and {group_col}**")

        p_interp = ResultInterpreter.interpret_p_value(p_value)
        st.metric("Result", f"{p_interp['significance']} {p_interp['emoji']}")

        st.markdown("### Distribution")
        st.dataframe(contingency)

        if p_interp["is_significant"]:
            st.success(f"✅ {outcome_col} distribution differs significantly across {group_col} groups")
        else:
            st.info(f"ℹ️ {outcome_col} distribution is similar across groups")


def run_predictor_analysis(df: pd.DataFrame, context: AnalysisContext):
    """Run regression to find predictors."""
    st.markdown("## 🎯 Finding Predictors")

    outcome_col = context.primary_variable
    predictors = context.predictor_variables

    # Prepare data
    analysis_cols = [outcome_col] + predictors
    analysis_df = df[analysis_cols].copy()

    # Check outcome type
    outcome_data = analysis_df[outcome_col].dropna()
    n_unique = outcome_data.nunique()

    if n_unique != 2:
        st.warning(f"Outcome has {n_unique} unique values. For now, only binary outcomes are supported.")
        return

    # Handle categorical predictors
    categorical_cols = analysis_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if outcome_col in categorical_cols:
        categorical_cols.remove(outcome_col)

    if categorical_cols:
        st.info(f"Converting categorical variables: {', '.join(categorical_cols)}")
        analysis_df = pd.get_dummies(analysis_df, columns=categorical_cols, drop_first=True)
        new_predictors = [c for c in analysis_df.columns if c != outcome_col]
    else:
        new_predictors = predictors

    # Drop missing
    analysis_df = analysis_df.dropna()

    if len(analysis_df) < 10:
        st.error("Need at least 10 complete observations")
        return

    # Run logistic regression
    model, summary_df = run_logistic_regression(analysis_df, outcome_col, new_predictors)

    st.markdown("### Results")
    st.markdown(f"**What predicts {outcome_col}?**")

    # Show significant predictors
    significant = summary_df[summary_df["P-Value"] < 0.05]

    if len(significant) > 0:
        st.success(f"✅ Found {len(significant)} significant predictor(s)")

        for var in significant.index:
            if var == "Intercept":
                continue

            or_val = summary_df.loc[var, "Odds Ratio"]
            p_val = summary_df.loc[var, "P-Value"]

            if or_val > 1:
                direction = "increases"
                pct = (or_val - 1) * 100
            else:
                direction = "decreases"
                pct = (1 - or_val) * 100

            st.markdown(f"**{var}** {direction} the odds by ~{pct:.0f}% (p={p_val:.4f})")

    else:
        st.info("ℹ️ No significant predictors found at p<0.05")

    # Full results in expander
    with st.expander("📊 Detailed Results"):
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


def run_survival_analysis(df: pd.DataFrame, context: AnalysisContext):
    """Run survival analysis."""
    st.markdown("## ⏱️ Survival Analysis")

    time_col = context.time_variable
    event_col = context.event_variable

    # Need to know which value means event occurred
    event_data = df[event_col].dropna()
    unique_vals = sorted(event_data.unique())

    if len(unique_vals) != 2:
        st.error(f"Event variable must be binary (has {len(unique_vals)} values)")
        return

    st.info(f"ℹ️ Event values: {unique_vals}")
    event_value = st.radio("Which value means the event occurred?", unique_vals, horizontal=True)

    # Prepare data
    analysis_df = df[[time_col, event_col]].copy()
    analysis_df[event_col] = (analysis_df[event_col] == event_value).astype(int)
    analysis_df = analysis_df.dropna()

    if len(analysis_df) < 10:
        st.error("Need at least 10 complete observations")
        return

    # Run Kaplan-Meier
    kmf, summary_df = run_kaplan_meier(
        analysis_df, duration_col=time_col, event_col=event_col, group_col=context.grouping_variable
    )

    st.markdown("### Results")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary_df["time"], summary_df["survival_probability"], linewidth=2)
    ax.fill_between(summary_df["time"], summary_df["ci_lower"], summary_df["ci_upper"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Survival Curve")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Median survival
    median_survival = kmf.median_survival_time_
    st.metric(
        "Median Survival Time",
        f"{median_survival:.1f}" if not np.isnan(median_survival) else "Not reached",
    )


def run_relationship_analysis(df: pd.DataFrame, context: AnalysisContext):
    """Explore relationships between variables."""
    st.markdown("## 🔗 Relationships Between Variables")

    variables = context.predictor_variables

    if len(variables) < 2:
        st.warning("Need at least 2 variables to examine relationships")
        return

    # Calculate correlations
    analysis_df = df[variables].dropna()

    if len(analysis_df) < 3:
        st.error("Need at least 3 observations")
        return

    corr_matrix = analysis_df.corr()

    st.markdown("### Correlation Heatmap")

    # Simple heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    import seaborn as sns

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("How Variables Relate")
    st.pyplot(fig)

    # Highlight strong correlations
    st.markdown("### Strong Relationships")

    found_strong = False
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                corr_val = corr_matrix.loc[var1, var2]
                if abs(corr_val) >= 0.5:
                    found_strong = True
                    direction = "positive" if corr_val > 0 else "negative"
                    strength = "strong" if abs(corr_val) >= 0.7 else "moderate"
                    st.markdown(f"**{var1}** and **{var2}**: {strength} {direction} relationship (r={corr_val:.2f})")

    if not found_strong:
        st.info("ℹ️ No strong correlations found (|r| >= 0.5)")


def main():
    st.title("💬 Ask Questions")
    st.markdown("""
    Ask questions about your data in plain English. I'll figure out the right analysis and explain the results.
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
            st.stop()

    # Show Semantic Scope in sidebar
    with st.sidebar.expander("🔍 Semantic Scope", expanded=False):
        st.markdown("**V1 Cohort-First Mode**")

        # Cohort table status
        st.markdown(f"✅ **Cohort Table**: {len(cohort):,} rows")

        # Multi-table status
        if MULTI_TABLE_ENABLED:
            st.markdown("⚠️ **Multi-Table**: Experimental")
        else:
            st.markdown("⏸️ **Multi-Table**: Disabled (V2)")

        # Detected grain
        grain = "patient_level"  # Default for V1
        st.markdown(f"📊 **Grain**: {grain}")

        # Outcome column (if detected)
        outcome_cols = [
            c for c in cohort.columns if "outcome" in c.lower() or "death" in c.lower() or "mortality" in c.lower()
        ]
        if outcome_cols:
            st.markdown(f"🎯 **Outcome**: `{outcome_cols[0]}`")
        else:
            st.markdown("🎯 **Outcome**: Not specified")

        # Show column count
        st.caption(f"{len(cohort.columns)} columns available")

    st.divider()

    # Initialize session state for context
    if "analysis_context" not in st.session_state:
        st.session_state["analysis_context"] = None
        st.session_state["intent_signal"] = None
        st.session_state["use_nl_query"] = True  # Default to NL query first

    # Step 1: Ask question (NL or structured)
    if st.session_state["intent_signal"] is None:
        # Try free-form NL query first
        if st.session_state["use_nl_query"]:
            try:
                # Get semantic layer using contract pattern
                semantic_layer = dataset.get_semantic_layer()
                context = QuestionEngine.ask_free_form_question(semantic_layer)

                if context:
                    # Successfully parsed NL query
                    st.session_state["analysis_context"] = context
                    st.session_state["intent_signal"] = "nl_parsed"
                    st.rerun()

            except ValueError:
                # Semantic layer not available
                st.info("Natural language queries are only available for datasets with semantic layers.")
                st.session_state["use_nl_query"] = False
                st.rerun()
                return
            except Exception as e:
                st.error(f"Error parsing natural language query: {e}")
                st.session_state["use_nl_query"] = False

            # Show option to use structured questions instead
            st.divider()
            st.markdown("### Or use structured questions")
            if st.button("💬 Use structured questions instead", help="Choose from predefined question types"):
                st.session_state["use_nl_query"] = False
                st.rerun()

        else:
            # Use structured questions
            intent_signal = QuestionEngine.ask_initial_question(cohort)

            if intent_signal:
                if intent_signal == "help":
                    st.divider()
                    help_answers = QuestionEngine.ask_help_questions(cohort)

                    # Map help answers to intent
                    if help_answers.get("has_time"):
                        intent_signal = "survival"
                    elif help_answers.get("has_outcome"):
                        intent_signal = help_answers.get("approach", "predict")
                    else:
                        intent_signal = "describe"

                st.session_state["intent_signal"] = intent_signal
                st.session_state["analysis_context"] = QuestionEngine.build_context_from_intent(intent_signal, cohort)
                st.rerun()

            # Show option to go back to NL query
            st.divider()
            if st.button("🔙 Try natural language query instead"):
                st.session_state["use_nl_query"] = True
                st.rerun()

    else:
        # We have intent, now gather details
        context = st.session_state["analysis_context"]

        st.divider()

        # Ask follow-up questions based on intent
        if context.inferred_intent == AnalysisIntent.DESCRIBE:
            # No additional questions needed for describe
            context.primary_variable = "all"

        elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
            if not context.primary_variable:
                context.primary_variable = QuestionEngine.select_primary_variable(
                    cohort, context, "What do you want to compare?"
                )

            if context.primary_variable and not context.grouping_variable:
                context.grouping_variable = QuestionEngine.select_grouping_variable(
                    cohort, exclude=[context.primary_variable]
                )

        elif context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
            if not context.primary_variable:
                context.primary_variable = QuestionEngine.select_primary_variable(
                    cohort, context, "What outcome do you want to predict?"
                )

            if context.primary_variable and not context.predictor_variables:
                context.predictor_variables = QuestionEngine.select_predictor_variables(
                    cohort, exclude=[context.primary_variable], min_vars=1
                )

        elif context.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
            if not context.time_variable or not context.event_variable:
                context.time_variable, context.event_variable = QuestionEngine.select_time_variables(cohort)

        elif context.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
            if len(context.predictor_variables) < 2:
                context.predictor_variables = QuestionEngine.select_predictor_variables(cohort, exclude=[], min_vars=2)

        # Update context in session state
        st.session_state["analysis_context"] = context

        # Show progress
        st.divider()
        QuestionEngine.render_progress_indicator(context)

        # If complete, show run button
        if context.is_complete_for_intent():
            if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
                st.divider()

                with st.spinner("Analyzing..."):
                    # Run appropriate analysis
                    if context.inferred_intent == AnalysisIntent.DESCRIBE:
                        run_descriptive_analysis(cohort, context)

                    elif context.inferred_intent == AnalysisIntent.COMPARE_GROUPS:
                        run_comparison_analysis(cohort, context)

                    elif context.inferred_intent == AnalysisIntent.FIND_PREDICTORS:
                        run_predictor_analysis(cohort, context)

                    elif context.inferred_intent == AnalysisIntent.EXAMINE_SURVIVAL:
                        run_survival_analysis(cohort, context)

                    elif context.inferred_intent == AnalysisIntent.EXPLORE_RELATIONSHIPS:
                        run_relationship_analysis(cohort, context)

        # Reset button
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state["analysis_context"] = None
            st.session_state["intent_signal"] = None
            st.rerun()


if __name__ == "__main__":
    main()
