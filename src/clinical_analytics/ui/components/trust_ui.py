"""
Trust UI Components (ADR003 Phase 1).

Provides trust verification expander showing:
- Raw QueryPlan (as parsed)
- Alias-Resolved Plan (canonical names)
- Effective Execution (dataset, entity_key, filters, cohort size)
- Patient-Level Export (capped, with download button)
"""

from typing import Any

import polars as pl

from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.semantic import SemanticLayer


class TrustUI:
    """Trust UI component for transparent verification and patient-level export."""

    @staticmethod
    def _extract_raw_fields(query_plan: QueryPlan) -> dict[str, Any]:
        """
        Extract raw QueryPlan fields for display.

        Args:
            query_plan: QueryPlan from NLU parsing

        Returns:
            Dict with raw intent, metric, group_by, filters
        """
        return {
            "intent": query_plan.intent,
            "metric": query_plan.metric,
            "group_by": query_plan.group_by,
            "filters": [
                {
                    "column": f.column,
                    "operator": f.operator,
                    "value": f.value,
                }
                for f in query_plan.filters
            ],
        }

    @staticmethod
    def _resolve_aliases(query_plan: QueryPlan, semantic_layer: SemanticLayer) -> dict[str, Any]:
        """
        Resolve aliases to canonical column names.

        Args:
            query_plan: QueryPlan with potential aliases
            semantic_layer: SemanticLayer with alias index

        Returns:
            Dict with resolved canonical names
        """
        alias_index = semantic_layer.get_column_alias_index()

        # Resolve metric
        resolved_metric = None
        if query_plan.metric:
            normalized = semantic_layer._normalize_alias(query_plan.metric)
            resolved_metric = alias_index.get(normalized, query_plan.metric)

        # Resolve group_by
        resolved_group_by = None
        if query_plan.group_by:
            normalized = semantic_layer._normalize_alias(query_plan.group_by)
            resolved_group_by = alias_index.get(normalized, query_plan.group_by)

        # Resolve filters
        resolved_filters = []
        for f in query_plan.filters:
            normalized = semantic_layer._normalize_alias(f.column)
            canonical_col = alias_index.get(normalized, f.column)
            resolved_filters.append(
                {
                    "column": canonical_col,
                    "operator": f.operator,
                    "value": f.value,
                }
            )

        return {
            "metric": resolved_metric,
            "group_by": resolved_group_by,
            "filters": resolved_filters,
        }

    @staticmethod
    def _compute_effective_execution(
        query_plan: QueryPlan,
        cohort: pl.DataFrame,
        dataset_version: str,
        entity_key: str | None,
    ) -> dict[str, Any]:
        """
        Compute effective execution details.

        Args:
            query_plan: QueryPlan to execute
            cohort: Cohort DataFrame
            dataset_version: Dataset version identifier
            entity_key: Entity key for COUNT (e.g., patient_id)

        Returns:
            Dict with effective execution details
        """
        # Normalize filters to detect tautologies
        effective_filters = TrustUI._normalize_effective_filters(query_plan.filters, cohort)

        # Calculate cohort size
        cohort_size = TrustUI._calculate_cohort_size(cohort, query_plan.filters, entity_key)

        return {
            "dataset_version": dataset_version,
            "entity_key": entity_key,
            "effective_filters": effective_filters,
            "cohort_size": cohort_size,
        }

    @staticmethod
    def _extract_audit_info(query_plan: QueryPlan, query_text: str) -> dict:
        """
        Extract audit trail info (run_key, query_text).

        Args:
            query_plan: QueryPlan with run_key
            query_text: Original query text

        Returns:
            Dict with run_key and query_text
        """
        return {
            "run_key": query_plan.run_key,
            "query_text": query_text,
        }

    @staticmethod
    def _prepare_patient_export(
        cohort: pl.DataFrame,
        query_plan: QueryPlan,
        max_rows: int | None = 100,
    ) -> pl.DataFrame:
        """
        Prepare patient-level export DataFrame.

        Args:
            cohort: Full cohort DataFrame
            query_plan: QueryPlan (for extracting filter columns)
            max_rows: Maximum rows to export (default: 100, None for full export)

        Returns:
            Export DataFrame with audit columns (patient_id, primary_variable, filter columns)
        """
        # Select audit columns: patient_id + primary_variable + filter columns
        export_cols = {"patient_id"}

        if query_plan.metric and query_plan.metric in cohort.columns:
            export_cols.add(query_plan.metric)

        for f in query_plan.filters:
            if f.column in cohort.columns:
                export_cols.add(f.column)

        # Select columns that exist in cohort
        available_cols = [col for col in export_cols if col in cohort.columns]

        export_df = cohort.select(available_cols)

        # Limit rows if max_rows specified
        if max_rows is not None:
            export_df = export_df.head(max_rows)

        return export_df

    @staticmethod
    def _calculate_cohort_size(
        cohort: pl.DataFrame,
        filters: list[FilterSpec],
        entity_key: str | None,
    ) -> dict[str, Any]:
        """
        Calculate cohort size (count_total, count_filtered, percentage).

        Args:
            cohort: Cohort DataFrame
            filters: Filter conditions
            entity_key: Entity key for counting (e.g., patient_id)

        Returns:
            Dict with count_total, count_filtered, percentage
        """
        # Count total (denominator): all rows with same entity_key
        if entity_key and entity_key in cohort.columns:
            count_total = cohort.select(pl.col(entity_key)).n_unique()
        else:
            count_total = cohort.height

        # Apply filters to get filtered cohort
        filtered_cohort = cohort
        for f in filters:
            if f.column not in filtered_cohort.columns:
                continue

            if f.operator == "==":
                filtered_cohort = filtered_cohort.filter(pl.col(f.column) == f.value)
            elif f.operator == "!=":
                filtered_cohort = filtered_cohort.filter(pl.col(f.column) != f.value)
            elif f.operator == ">":
                filtered_cohort = filtered_cohort.filter(pl.col(f.column) > f.value)
            elif f.operator == ">=":
                filtered_cohort = filtered_cohort.filter(pl.col(f.column) >= f.value)
            elif f.operator == "<":
                filtered_cohort = filtered_cohort.filter(pl.col(f.column) < f.value)
            elif f.operator == "<=":
                filtered_cohort = filtered_cohort.filter(pl.col(f.column) <= f.value)
            elif f.operator == "IN":
                # Ensure value is a list for is_in()
                values = f.value if isinstance(f.value, list) else [f.value]
                filtered_cohort = filtered_cohort.filter(pl.col(f.column).is_in(values))
            elif f.operator == "NOT_IN":
                # Ensure value is a list for is_in()
                values = f.value if isinstance(f.value, list) else [f.value]
                filtered_cohort = filtered_cohort.filter(~pl.col(f.column).is_in(values))

        # Count filtered (numerator)
        if entity_key and entity_key in filtered_cohort.columns:
            count_filtered = filtered_cohort.select(pl.col(entity_key)).n_unique()
        else:
            count_filtered = filtered_cohort.height

        # Calculate percentage
        percentage = (count_filtered / count_total * 100) if count_total > 0 else 0.0

        return {
            "count_total": count_total,
            "count_filtered": count_filtered,
            "percentage": percentage,
        }

    @staticmethod
    def _normalize_effective_filters(
        filters: list[FilterSpec],
        cohort: pl.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Normalize filters and detect tautologies.

        Args:
            filters: Filter conditions
            cohort: Cohort DataFrame (for detecting tautologies)

        Returns:
            List of normalized filter dicts with tautology detection
        """
        normalized_filters = []

        for f in filters:
            is_tautology = False

            # Detect tautologies (filters that don't actually restrict the data)
            if f.operator == "IN" and f.column in cohort.columns:
                # Check if filter values cover all unique values in the column
                unique_values = cohort.select(pl.col(f.column)).unique().to_series().to_list()
                # Ensure value is a list for set operations
                filter_values = f.value if isinstance(f.value, list) else [f.value]
                if set(unique_values).issubset(set(filter_values)):
                    is_tautology = True

            normalized_filters.append(
                {
                    "column": f.column,
                    "operator": f.operator,
                    "value": f.value,
                    "is_tautology": is_tautology,
                    "label": "non-restrictive" if is_tautology else "active",
                }
            )

        return normalized_filters

    @staticmethod
    def render_verification(
        query_plan: QueryPlan,
        result: dict,
        cohort: pl.DataFrame,
        dataset_version: str,
        query_text: str = "",
        semantic_layer: SemanticLayer | None = None,
    ) -> None:
        """
        Render trust verification expander (Streamlit UI).

        Args:
            query_plan: QueryPlan from NLU parsing
            result: Analysis result dict
            cohort: Cohort DataFrame
            dataset_version: Dataset version identifier
            query_text: Original query text
            semantic_layer: Optional SemanticLayer with alias resolution (Phase 2)
        """
        import streamlit as st

        with st.expander("🔎 Verify: Show source patients"):
            # Section 1: Raw QueryPlan
            st.markdown("### Raw QueryPlan")
            raw_fields = TrustUI._extract_raw_fields(query_plan)
            st.json(raw_fields)

            # Section 2: Alias-Resolved Plan (Phase 2 - requires semantic layer)
            if semantic_layer:
                st.markdown("### Alias-Resolved Plan")
                resolved_plan = TrustUI._resolve_aliases(query_plan, semantic_layer)
                st.json(resolved_plan)

            # Section 3: Effective Execution
            st.markdown("### Effective Execution")
            entity_key = "patient_id"  # Default entity key
            effective_execution = TrustUI._compute_effective_execution(query_plan, cohort, dataset_version, entity_key)

            st.write(f"**Dataset**: {effective_execution['dataset_version']}")
            st.write(f"**Entity Key**: {effective_execution['entity_key']}")

            # Show effective filters
            st.markdown("**Effective Filters:**")
            for ef in effective_execution["effective_filters"]:
                label = ef["label"]
                st.write(f"- `{ef['column']}` {ef['operator']} `{ef['value']}` ({label})")

            # Show cohort size
            cohort_size = effective_execution["cohort_size"]
            filtered = cohort_size["count_filtered"]
            total = cohort_size["count_total"]
            pct = cohort_size["percentage"]
            st.write(f"**Cohort Size**: {filtered:,} / {total:,} ({pct:.1f}%)")

            # Section 4: Run-Key and Audit Trail
            st.markdown("### Audit Trail")
            audit_info = TrustUI._extract_audit_info(query_plan, query_text)
            st.write(f"**Run Key**: `{audit_info['run_key']}`")
            st.write(f"**Generated from**: {audit_info['query_text']}")

            # Section 5: Patient-Level Export
            st.markdown("### Patient-Level Export")
            export_df = TrustUI._prepare_patient_export(cohort, query_plan, max_rows=100)

            st.write(f"Showing {export_df.height} rows (capped at 100)")
            # PANDAS EXCEPTION: Required for Streamlit st.dataframe display
            # TODO: Remove when Streamlit supports Polars natively
            st.dataframe(export_df.to_pandas())

            # Download button for full export
            if cohort.height > 100:
                full_export = TrustUI._prepare_patient_export(cohort, query_plan, max_rows=None)
                csv_data = full_export.write_csv()
                st.download_button(
                    label="Download Full Export",
                    data=csv_data,
                    file_name="patient_export.csv",
                    mime="text/csv",
                )
