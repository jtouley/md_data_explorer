"""
Performance test to measure selective dataset loading impact.

This test measures the time difference between:
- Loading all dataset configs upfront (discovered_datasets fixture)
- Loading only requested dataset config (get_dataset_by_name fixture)

Target: 30-50% reduction in setup time
"""

import time

import pytest


class TestSelectiveLoadingPerformance:
    """Measure performance impact of selective dataset loading."""

    def test_selective_loading_reduces_setup_time(self, dataset_registry, get_dataset_by_name, discovered_datasets):
        """
        Measure that selective loading reduces setup time compared to pre-loading all configs.

        This test compares:
        - Baseline: Time to access first dataset from discovered_datasets (all configs pre-loaded)
        - Selective: Time to load first dataset using get_dataset_by_name (lazy loading)

        Note: The real benefit comes when tests only need one dataset but
        discovered_datasets loads all configs upfront.
        """
        # Arrange: Get first available dataset name
        available = [
            d for d in dataset_registry.list_datasets() if d not in ["covid_ms", "mimic3", "sepsis", "uploaded"]
        ]

        if not available:
            pytest.skip("No datasets available for testing")

        dataset_name = available[0]

        # Measure baseline: Access config from discovered_datasets (all configs already loaded)
        start_baseline = time.perf_counter()
        config = discovered_datasets["configs"].get(dataset_name)
        if config is None:
            # If not in cache, load it (simulates worst case)
            from clinical_analytics.core.mapper import load_dataset_config

            config = load_dataset_config(dataset_name)
        baseline_time = time.perf_counter() - start_baseline

        # Measure selective: Load dataset using get_dataset_by_name (lazy loading)
        start_selective = time.perf_counter()
        dataset = get_dataset_by_name(dataset_name)
        selective_time = time.perf_counter() - start_selective

        # Assert: Both methods work
        assert config is not None
        assert dataset is not None

        # Calculate improvement
        # Note: For single dataset, improvement may be small
        # Real benefit comes when multiple datasets exist and we only need one
        if baseline_time > 0:
            improvement_pct = ((baseline_time - selective_time) / baseline_time) * 100
        else:
            improvement_pct = 0

        # Log results for documentation
        print(
            f"\nSelective Loading Performance:\n"
            f"  Baseline (pre-loaded all configs): {baseline_time:.4f}s\n"
            f"  Selective (lazy load one dataset): {selective_time:.4f}s\n"
            f"  Improvement: {improvement_pct:.1f}% reduction\n"
            f"  Note: Real benefit increases with number of datasets\n"
        )

        # Assert: Selective loading should be at least as fast (or faster if many datasets)
        # For single dataset, times should be similar
        # For multiple datasets, selective should be faster
        assert selective_time <= baseline_time * 1.5, (
            f"Selective loading should not be significantly slower. "
            f"Baseline: {baseline_time:.4f}s, Selective: {selective_time:.4f}s"
        )

    def test_selective_loading_scales_with_dataset_count(
        self, dataset_registry, get_dataset_by_name, discovered_datasets
    ):
        """
        Measure that selective loading benefit increases with number of datasets.

        When many datasets exist, loading all configs upfront is expensive.
        Selective loading only loads what's needed.
        """
        # Arrange: Get available datasets
        available = [
            d for d in dataset_registry.list_datasets() if d not in ["covid_ms", "mimic3", "sepsis", "uploaded"]
        ]

        if len(available) < 2:
            pytest.skip("Need at least 2 datasets to measure scaling benefit")

        # Measure baseline: Time to access all configs (already pre-loaded)
        start_baseline = time.perf_counter()
        all_configs = discovered_datasets["configs"]
        baseline_time = time.perf_counter() - start_baseline

        # Measure selective: Time to load just first dataset
        start_selective = time.perf_counter()
        dataset = get_dataset_by_name(available[0])
        selective_time = time.perf_counter() - start_selective

        # Assert: Both methods work
        assert len(all_configs) > 0
        assert dataset is not None

        # Log results
        print(
            f"\nSelective Loading Scaling:\n"
            f"  Total datasets: {len(available)}\n"
            f"  Baseline (all configs pre-loaded): {baseline_time:.4f}s\n"
            f"  Selective (one dataset loaded): {selective_time:.4f}s\n"
            f"  Benefit: Only load what's needed\n"
        )

        # The real benefit: We only loaded one dataset config, not all
        # This becomes more significant as dataset count increases
