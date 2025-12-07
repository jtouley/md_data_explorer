import polars as pl
from pathlib import Path

def load_raw_data(path: Path) -> pl.DataFrame:
    """Read the raw CSV file using Polars for efficient processing."""
    if not path.exists():
        raise FileNotFoundError(f"COVID-MS data not found at {path}")

    return pl.read_csv(path)

def normalize_outcome(val: str) -> int:
    """Normalize outcome values to binary integers."""
    if val is None:
        return 0
    val = val.lower().strip()
    if val == 'yes':
        return 1
    if val == 'no':
        return 0
    return 0

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and standardize the COVID-MS dataframe using Polars.

    1. Normalize hospitalization outcomes
    2. Standardize sex column
    3. Handle missing data
    """
    # Use Polars expressions for efficient data cleaning
    df = df.with_columns([
        # Normalize outcomes using when-then-otherwise (Polars equivalent of apply)
        pl.when(pl.col('covid19_admission_hospital').str.to_lowercase() == 'yes')
          .then(1)
          .otherwise(0)
          .alias('outcome_hospitalized'),

        pl.when(pl.col('covid19_icu_stay').str.to_lowercase() == 'yes')
          .then(1)
          .otherwise(0)
          .alias('outcome_icu'),

        pl.when(pl.col('covid19_ventilation').str.to_lowercase() == 'yes')
          .then(1)
          .otherwise(0)
          .alias('outcome_ventilation'),

        # Standardize Sex with fill_null
        pl.col('sex').fill_null('Unknown').cast(pl.Utf8).alias('sex')
    ])

    return df

