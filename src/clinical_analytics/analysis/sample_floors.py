# SPDX-License-Identifier: MIT
"""Smallest samples each analysis runs on.

compute.py refuses below these: a statistic on fewer rows is not defined or not meaningful
(a t-test needs two values per group, a logistic model needs enough rows to fit).
"""

from typing import Final

MIN_COMPARISON_N: Final = 2
MIN_GROUP_N: Final = 2  # per group, t-test and ANOVA
MIN_REGRESSION_N: Final = 10
MIN_SURVIVAL_N: Final = 10
MIN_CORRELATION_N: Final = 3
