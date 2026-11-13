# SPDX-License-Identifier: MIT
"""Smallest samples each analysis runs on.

compute.py refuses below these, so a reader who sees an answer knows the sample was at least
this large; result_presentation.py uses the same numbers when it decides what a hidden count
would give away.
"""

from typing import Final

MIN_COMPARISON_N: Final = 2
MIN_GROUP_N: Final = 2  # per group, t-test and ANOVA
MIN_REGRESSION_N: Final = 10
MIN_SURVIVAL_N: Final = 10
MIN_CORRELATION_N: Final = 3
