import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning


class ModelNotEstimableError(ValueError):
    """The model cannot produce valid estimates for this data (e.g. perfect separation)."""


_QUOTED_TERM = re.compile(r'Q\("((?:[^"\\]|\\.)*)"\)')


def _quote(name: str) -> str:
    """Patsy-quote a column so names with spaces, colons or punctuation parse."""
    return 'Q("' + name.replace("\\", "\\\\").replace('"', '\\"') + '")'


def _term_label(term: str) -> str:
    """Strip patsy quoting from a coefficient name: 'Q("Systolic BP")' -> 'Systolic BP'."""
    return _QUOTED_TERM.sub(lambda m: m.group(1).replace('\\"', '"'), term)


def run_logistic_regression(df: pd.DataFrame, outcome_col: str, predictors: list[str]) -> tuple[Any, pd.DataFrame]:
    """
    Run logistic regression on a cohort dataframe.

    Args:
        df: Dataframe containing outcome and predictors. The outcome must already be 0/1.
        outcome_col: Name of the binary outcome column.
        predictors: List of column names to use as independent variables.

    Returns:
        model: The fitted statsmodels LogitResult object.
        summary_df: A clean dataframe of Odds Ratios, CI, and P-values, indexed by column name.

    Raises:
        ValueError: No rows remain after dropping nulls.
        ModelNotEstimableError: Perfect separation or a fit that did not converge, so the odds
            ratios would be meaningless.
    """
    model_cols = [outcome_col] + predictors
    data = df[model_cols].dropna()

    if data.empty:
        raise ValueError("No data remaining after dropping nulls.")

    design = np.column_stack([np.ones(len(data)), data[predictors].to_numpy(dtype=float)])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ModelNotEstimableError(
            "Some predictors are exact combinations of others (perfect collinearity), so their odds ratios "
            f"cannot be separated: {', '.join(predictors)}."
        )

    formula = f"{_quote(outcome_col)} ~ {' + '.join(_quote(p) for p in predictors)}"

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        try:
            model = smf.logit(formula=formula, data=data).fit(disp=0)
        except (PerfectSeparationError, PerfectSeparationWarning) as exc:
            raise ModelNotEstimableError(
                f"Perfect separation: a predictor perfectly predicts {outcome_col}, so odds ratios cannot be estimated."
            ) from exc

    params = model.params
    # A large intercept is normal (e.g. a predictor like calendar year); only slopes signal separation.
    slopes = params.drop("Intercept", errors="ignore")
    if not model.mle_retvals.get("converged", True) or bool((slopes.abs() > 20).any()):
        raise ModelNotEstimableError(
            f"The model for {outcome_col} did not converge to finite estimates (likely near-perfect separation)."
        )
    conf = model.conf_int()
    conf.columns = ["CI Lower", "CI Upper"]
    pvalues = model.pvalues

    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)

    summary_df = pd.DataFrame(
        {
            "Odds Ratio": odds_ratios,
            "CI Lower": conf_or["CI Lower"],
            "CI Upper": conf_or["CI Upper"],
            "P-Value": pvalues,
        }
    )
    summary_df.index = [_term_label(str(term)) for term in summary_df.index]

    return model, summary_df
