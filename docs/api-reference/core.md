# Core API Reference

## Semantic Layer

::: clinical_analytics.core.semantic.SemanticLayer
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - get_base_view
        - get_available_metrics
        - get_available_dimensions
        - get_outcome_df
        - compile_to_sql

## Dataset Registry

::: clinical_analytics.core.registry.DatasetRegistry
    options:
      show_root_heading: true
      show_source: true
      members:
        - register_dataset
        - get_dataset
        - list_datasets
        - get_semantic_layer

## Natural Language Query Engine

*Coming soon in Phase 2*

The NL Query Engine will parse natural language queries into structured intents.

## Schema Inference

*Coming soon in Phase 3*

Automatic schema detection for uploaded datasets.

## Multi-Table Handler

*Coming soon in Phase 4*

Support for complex datasets with multiple related tables.

## Storage

::: clinical_analytics.storage.datastore.DataStore
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - save_table
        - load_table
        - list_datasets
        - export_to_parquet

::: clinical_analytics.storage.versioning
    options:
      show_root_heading: true
      show_source: true

::: clinical_analytics.storage.query_logger.QueryLogger
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - log_query
        - log_execution
        - log_result
        - log_follow_up