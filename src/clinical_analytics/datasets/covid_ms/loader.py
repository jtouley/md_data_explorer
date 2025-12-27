from pathlib import Path

import polars as pl

from clinical_analytics.core.mapper import ColumnMapper


def load_raw_data(path: Path) -> pl.DataFrame:
    """Read the raw CSV file using Polars for efficient processing."""
    if not path.exists():
        raise FileNotFoundError(f"COVID-MS data not found at {path}")

    return pl.read_csv(path)


def clean_data(df: pl.DataFrame, mapper: ColumnMapper | None = None) -> pl.DataFrame:
    """
    Clean and standardize the COVID-MS dataframe using Polars.

    Now config-driven: Outcome transformations are applied via mapper.
    Only basic data cleaning (null handling) is done here.

    Args:
        df: Raw Polars DataFrame
        mapper: ColumnMapper instance for config-driven transformations

    Returns:
        Cleaned DataFrame with outcome transformations applied
    """
    # Basic data cleaning: handle nulls in sex column
    df = df.with_columns([pl.col("sex").fill_null("Unknown").cast(pl.Utf8).alias("sex")])

    # Apply outcome transformations via mapper (config-driven)
    if mapper:
        df = mapper.apply_outcome_transformations(df)

    return df
