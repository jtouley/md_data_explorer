from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def run_logistic_regression(df: pd.DataFrame, outcome_col: str, predictors: list[str]) -> tuple[Any, pd.DataFrame]:
    """
    Run logistic regression on a cohort dataframe.

    Args:
        df: Dataframe containing outcome and predictors.
        outcome_col: Name of the binary outcome column.
        predictors: List of column names to use as independent variables.

    Returns:
        model: The fitted statsmodels LogitResult object.
        summary_df: A clean dataframe of Odds Ratios, CI, and P-values.
    """
    # clean data
    model_cols = [outcome_col] + predictors
    data = df[model_cols].dropna()

    if data.empty:
        raise ValueError("No data remaining after dropping nulls.")

    # Construct formula
    # Quote column names to handle spaces or special chars if any
    # (though best practice is clean names)
    formula = f"{outcome_col} ~ {' + '.join(predictors)}"

    # Fit model
    model = smf.logit(formula=formula, data=data).fit(disp=0)

    # Extract results
    params = model.params
    conf = model.conf_int()
    conf.columns = ["CI Lower", "CI Upper"]
    pvalues = model.pvalues

    # Calculate Odds Ratios (vectorized)
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)

    # Combine into summary
    summary_df = pd.DataFrame(
        {
            "Odds Ratio": odds_ratios,
            "CI Lower": conf_or["CI Lower"],
            "CI Upper": conf_or["CI Upper"],
            "P-Value": pvalues,
        }
    )

    # Format
    summary_df = summary_df.round(4)

    return model, summary_df
