"""
Correlation Analysis Page

Explore relationships between numeric variables using correlation matrices.
Understand how variables relate to each other.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Keep only lightweight imports at module scope
from clinical_analytics.core.schema import UnifiedCohort

# Heavy imports moved inside main() after gate

# Page config
st.set_page_config(page_title="Correlation Analysis | Clinical Analytics", page_icon="🔗", layout="wide")


def calculate_correlations(df: pd.DataFrame, method: str = "pearson") -> tuple:
    """
    Calculate correlation matrix and p-values.

    Args:
        df: DataFrame with numeric columns
        method: Correlation method ('pearson' or 'spearman')

    Returns:
        Tuple of (correlation matrix, p-value matrix)
    """
    # Calculate correlations
    if method == "pearson":
        corr_matrix = df.corr(method="pearson")
    else:
        corr_matrix = df.corr(method="spearman")

    # Calculate p-values
    n_vars = len(df.columns)
    p_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)), columns=df.columns, index=df.columns)

    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i != j:
                if method == "pearson":
                    _, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                else:
                    _, p_val = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                p_matrix.iloc[i, j] = p_val
            else:
                p_matrix.iloc[i, j] = 0.0  # Diagonal

    return corr_matrix, p_matrix


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame, p_matrix: pd.DataFrame, significance_level: float = 0.05
) -> plt.Figure:
    """
    Create correlation heatmap with significance markers.

    Args:
        corr_matrix: Correlation coefficient matrix
        p_matrix: P-value matrix
        significance_level: Alpha level for significance

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )

    # Add significance markers (stars)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            if i != j:  # Skip diagonal
                p_val = p_matrix.iloc[i, j]
                if p_val < 0.001:
                    marker = "***"
                elif p_val < 0.01:
                    marker = "**"
                elif p_val < significance_level:
                    marker = "*"
                else:
                    marker = ""

                if marker:
                    ax.text(
                        j + 0.5,
                        i + 0.7,
                        marker,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=10,
                        fontweight="bold",
                    )

    ax.set_title(
        "Correlation Matrix\n(* p<0.05, ** p<0.01, *** p<0.001)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    return fig


def main():
    # Gate: V1 MVP mode disables legacy pages
    # MUST run before any expensive operations
    from clinical_analytics.ui.helpers import gate_v1_mvp_legacy_page

    gate_v1_mvp_legacy_page()  # Stops execution if gated

    # NOW do heavy imports (after gate)
    from clinical_analytics.core.registry import DatasetRegistry
    from clinical_analytics.datasets.uploaded.definition import UploadedDatasetFactory
    from clinical_analytics.ui.components.result_interpreter import ResultInterpreter

    st.title("🔗 Explore Relationships")
    st.markdown("""
    Discover how variables relate to each other using **correlation analysis**.
    See which variables move together (positive correlation) or in opposite directions
    (negative correlation).
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
    # Check if this is an uploaded dataset (multiple checks for robustness)
    is_uploaded = (
        dataset_choice in uploaded_datasets
        or dataset_choice_display.startswith("📤")
        or dataset_choice not in available_datasets
    )

    # Load dataset
    with st.spinner(f"Loading {dataset_choice_display}..."):
        try:
            if is_uploaded:
                # For uploaded datasets, use the factory (requires upload_id)
                dataset = UploadedDatasetFactory.create_dataset(dataset_choice)
                dataset.load()
            else:
                # For built-in datasets, use the registry
                dataset = DatasetRegistry.get_dataset(dataset_choice)
                dataset.validate()
                dataset.load()

            cohort = dataset.get_cohort()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return

    # Configuration
    st.markdown("## 🔧 Configure Analysis")

    # Get numeric columns
    numeric_cols = cohort.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [UnifiedCohort.PATIENT_ID]]

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric variables for correlation analysis.")
        st.info(f"Found {len(numeric_cols)} numeric variable(s). Please upload data with more numeric variables.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Select Variables")
        selected_vars = st.multiselect(
            "Which variables do you want to correlate?",
            numeric_cols,
            default=numeric_cols[: min(8, len(numeric_cols))],
            help="Select 2-15 numeric variables to analyze relationships",
        )

        if len(selected_vars) < 2:
            st.warning("Please select at least 2 variables")
        elif len(selected_vars) > 15:
            st.warning("Too many variables selected. Maximum 15 for clear visualization.")

    with col2:
        st.markdown("### Correlation Method")
        corr_method = st.radio(
            "Choose method:",
            ["Pearson (for linear relationships)", "Spearman (for monotonic relationships)"],
            help=(
                "Pearson: measures linear relationships. Spearman: measures monotonic "
                "relationships (more robust to outliers)"
            ),
        )

        method = "pearson" if "Pearson" in corr_method else "spearman"

        st.markdown("### Significance Level")
        alpha = st.select_slider(
            "Alpha level:",
            options=[0.01, 0.05, 0.10],
            value=0.05,
            format_func=lambda x: (f"{x} ({'99%' if x == 0.01 else '95%' if x == 0.05 else '90%'} confidence)"),
        )

    # Data preview
    with st.expander("👀 Preview Data"):
        if selected_vars and len(selected_vars) >= 2:
            preview_df = cohort[selected_vars].dropna()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Complete Cases", len(preview_df))
            with col2:
                st.metric("Variables", len(selected_vars))
            with col3:
                missing_pct = (cohort[selected_vars].isna().sum().sum() / cohort[selected_vars].size) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")

            st.dataframe(preview_df.head(10))

    # Run analysis
    if not selected_vars or len(selected_vars) < 2:
        st.info("👆 Select at least 2 variables to continue")
        return

    if st.button("🔗 Analyze Correlations", type="primary"):
        with st.spinner("Calculating correlations..."):
            try:
                # Prepare data
                analysis_df = cohort[selected_vars].copy()

                # Drop missing values
                initial_n = len(analysis_df)
                analysis_df = analysis_df.dropna()
                final_n = len(analysis_df)

                if final_n < initial_n:
                    st.warning(
                        f"Dropped {initial_n - final_n} rows with missing data "
                        f"({(initial_n - final_n) / initial_n * 100:.1f}%)"
                    )

                if final_n < 3:
                    st.error("Need at least 3 complete observations for correlation analysis")
                    return

                # Calculate correlations
                corr_matrix, p_matrix = calculate_correlations(analysis_df, method=method)

                st.success("✅ Analysis complete!")

                # Display results
                st.markdown("## 📊 Results")

                # Heatmap
                st.markdown("### Correlation Heatmap")
                fig = plot_correlation_heatmap(corr_matrix, p_matrix, significance_level=alpha)
                st.pyplot(fig)

                st.caption("* p<0.05, ** p<0.01, *** p<0.001 (significant correlations)")

                # Correlation table
                st.markdown("### Correlation Coefficients")

                # Create detailed table
                correlation_details = []

                for i, var1 in enumerate(selected_vars):
                    for j, var2 in enumerate(selected_vars):
                        if i < j:  # Only upper triangle to avoid duplicates
                            corr_val = corr_matrix.loc[var1, var2]
                            p_val = p_matrix.loc[var1, var2]

                            # Interpret strength
                            abs_corr = abs(corr_val)
                            if abs_corr >= 0.7:
                                strength = "Strong"
                            elif abs_corr >= 0.5:
                                strength = "Moderate"
                            elif abs_corr >= 0.3:
                                strength = "Weak"
                            else:
                                strength = "Very Weak"

                            direction = "Positive" if corr_val > 0 else "Negative"

                            correlation_details.append(
                                {
                                    "Variable 1": var1,
                                    "Variable 2": var2,
                                    "Correlation": corr_val,
                                    "P-Value": p_val,
                                    "Strength": strength,
                                    "Direction": direction,
                                    "Significant": "✅" if p_val < alpha else "❌",
                                }
                            )

                details_df = pd.DataFrame(correlation_details)
                details_df = details_df.sort_values("Correlation", key=abs, ascending=False)

                st.dataframe(details_df.style.format({"Correlation": "{:.3f}", "P-Value": "{:.4f}"}))

                # Highlight significant correlations
                st.markdown("### 🔍 Significant Correlations")

                significant_corrs = details_df[details_df["P-Value"] < alpha]

                if len(significant_corrs) == 0:
                    st.info("No significant correlations found at the chosen significance level.")
                else:
                    st.success(f"Found {len(significant_corrs)} significant correlation(s)")

                    for idx, row in significant_corrs.iterrows():
                        var1 = row["Variable 1"]
                        var2 = row["Variable 2"]
                        corr_val = row["Correlation"]
                        p_val = row["P-Value"]

                        interpretation = ResultInterpreter.interpret_correlation(
                            correlation=corr_val, p_value=p_val, var1=var1, var2=var2
                        )

                        with st.expander(f"**{var1}** ↔ **{var2}** (r={corr_val:.3f})"):
                            st.markdown(interpretation)

                # Summary statistics
                with st.expander("📊 Summary Statistics"):
                    st.markdown("### Descriptive Statistics for Selected Variables")
                    desc_stats = analysis_df.describe().T
                    desc_stats["missing"] = initial_n - analysis_df.count()
                    st.dataframe(
                        desc_stats.style.format(
                            {
                                "mean": "{:.2f}",
                                "std": "{:.2f}",
                                "min": "{:.2f}",
                                "25%": "{:.2f}",
                                "50%": "{:.2f}",
                                "75%": "{:.2f}",
                                "max": "{:.2f}",
                            }
                        )
                    )

                # Export options
                st.markdown("## 📥 Export")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Save heatmap
                    from io import BytesIO

                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    buf.seek(0)

                    st.download_button(
                        "Download Heatmap",
                        buf,
                        f"correlation_heatmap_{dataset_choice}.png",
                        "image/png",
                    )

                with col2:
                    # Export correlation matrix
                    csv_data = corr_matrix.to_csv()
                    st.download_button(
                        "Download Correlation Matrix",
                        csv_data,
                        f"correlations_{dataset_choice}.csv",
                        "text/csv",
                    )

                with col3:
                    # Export detailed results
                    details_csv = details_df.to_csv(index=False)
                    st.download_button(
                        "Download Detailed Results",
                        details_csv,
                        f"correlation_details_{dataset_choice}.csv",
                        "text/csv",
                    )

                # Methods text
                with st.expander("📝 Methods Section Text"):
                    methods_text = ResultInterpreter.generate_methods_text(
                        analysis_type="correlation",
                        test_name=f"{method.capitalize()} correlation analysis",
                        variables={},
                    )

                    st.markdown(methods_text)

                    st.download_button(
                        "Download Methods Text",
                        methods_text,
                        "methods_correlation.txt",
                        "text/plain",
                    )

            except Exception as e:
                st.error(f"Error running analysis: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
