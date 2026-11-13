# SPDX-License-Identifier: MIT
"""Find a column that mixes the two units of one variable (DQ M3).

Glucose recorded partly in mg/dL and partly in mmol/L has no meaningful mean ("Mean glucose was
35.61", an average of two scales). Only variables with a configured unit pair are checked
(``contracts.config.UNIT_PAIRS``); a column is mixed when all of these hold:

* a two-component normal mixture fitted to its log values separates cleanly (Ashman's D above
  ``UNIT_MIX_MIN_SEPARATION``). A single distribution, however wide or skewed, fits two
  overlapping components, so plain wide variation is not a mix;
* two components explain the values much better than one (BIC gain of at least
  ``UNIT_MIX_MIN_BIC_GAIN``);
* the components' centres differ by the unit factor (within ``UNIT_MIX_RATIO_TOLERANCE``, and
  within ``UNIT_MIX_RATIO_TOLERANCE_SHARE`` of the factor for the small factors);
* each cluster holds at least ``UNIT_MIX_MIN_CLUSTER_VALUES`` values, and nearly all of it
  (``UNIT_MIX_IN_RANGE_SHARE``) is plausible in its own unit.

The upload report and the answer checks both call ``detect_unit_mix``. The result holds counts
and a cut point only. No patient value leaves this module: the cut point is a round number
strictly between the clusters, never a value from the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from clinical_analytics.contracts import config
from clinical_analytics.contracts.builder import find_plausibility_range, numbers_with_counts, plausibility_key
from clinical_analytics.contracts.declarations import UnitRule


def unit_pair_for(column: str) -> config.UnitPair | None:
    """The unit pair of ``column``: by its plausibility range (glucose, creatinine, temperature),
    else by the pair's own aliases (weight, height), a unit suffix allowed ("weight_kg")."""
    found = find_plausibility_range(column)
    key = plausibility_key(column)
    for pair in config.UNIT_PAIRS:
        if found is not None and found.variable == pair.variable:
            return pair
        if pair.aliases and _alias_of(key, _aliases(pair), pair) is not None:
            return pair
    return None


def _aliases(pair: config.UnitPair) -> frozenset[str]:
    found = next((r for r in config.PLAUSIBILITY_RANGES if r.variable == pair.variable), None)
    return pair.aliases | (found.aliases | found.hinted_aliases if found is not None else frozenset())


def _alias_of(key: str, aliases: frozenset[str], pair: config.UnitPair) -> str | None:
    """The unit a header names after an alias ("glucosemgdl" -> "mg/dL"), "" for a bare alias,
    None when ``key`` is not an alias of the pair."""
    for unit, hints in ((pair.low.unit, pair.low_hints), (pair.high.unit, pair.high_hints)):
        if any(key.endswith(hint) and key[: -len(hint)] in aliases for hint in hints):
            return unit
    return "" if key in aliases else None


def header_unit(column: str, pair: config.UnitPair) -> str | None:
    """The unit ``column``'s header names ("Glucose (mg/dL)" -> "mg/dL"), or None."""
    return _alias_of(plausibility_key(column), _aliases(pair), pair) or None


@dataclass(frozen=True)
class UnitMix:
    """A column read as two clusters, one per unit of ``pair``.

    ``low_count``/``high_count``: rows read as the low/high unit (every number the check read is
    in one of them, values at or below zero in the low one). ``cutoff``: a round number strictly
    between the clusters, or None when they overlap (some value's unit cannot be told with
    ``UNIT_MIX_CLEAN_POSTERIOR`` certainty, or the empty stretch between them is narrower than
    ``UNIT_MIX_CLEAN_GAP_SHARE`` of the units' log distance). ``target``: the unit the header names, else
    the larger cluster's (the high unit on a tie).
    """

    column: str
    pair: config.UnitPair
    low_count: int
    high_count: int
    cutoff: float | None
    target: str

    @property
    def present(self) -> int:
        return self.low_count + self.high_count

    @property
    def other(self) -> str:
        return self.pair.low.unit if self.target == self.pair.high.unit else self.pair.high.unit

    def rule(self, dataset_version: str | None) -> UnitRule | None:
        """The conversion the doctor may confirm: values on the other unit's side of the cut
        point are read in that unit and converted to ``target``. None when the clusters overlap
        (no cut point separates them)."""
        if self.cutoff is None:
            return None
        pair = self.pair
        to_high = self.target == pair.high.unit
        return UnitRule(
            column=self.column,
            cutoff=self.cutoff,
            side="below" if to_high else "at_or_above",
            source_unit=self.other,
            target_unit=self.target,
            scale=pair.scale if to_high else 1 / pair.scale,
            offset=pair.offset if to_high else -pair.offset / pair.scale,
            relation=relation_text(pair),
            dataset_version=dataset_version,
        )


def relation_text(pair: config.UnitPair) -> str:
    """'1 mmol/L = 18.016 mg/dL', or '°F = °C × 1.8 + 32' for a scale with an offset."""
    if pair.offset:
        return f"{pair.high.unit} = {pair.low.unit} × {pair.scale:g} + {pair.offset:g}"
    return f"1 {pair.low.unit} = {pair.scale:g} {pair.high.unit}"


def _normal(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def _otsu_split(x: np.ndarray, w: np.ndarray) -> int:
    """Index of the last low value of sorted ``x`` that maximises the between-cluster variance."""
    cw = np.cumsum(w)[:-1]
    cs = np.cumsum(w * x)[:-1]
    total_w, total_s = float(w.sum()), float((w * x).sum())
    m0, m1 = cs / cw, (total_s - cs) / (total_w - cw)
    return int(np.argmax(cw * (total_w - cw) * (m0 - m1) ** 2))


def _mixture(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(weights, means, spreads, log-likelihood) of a two-component normal mixture fitted to
    sorted ``x`` (counts ``w``) by EM, started from the Otsu split; component 0 has the lower
    mean."""
    floor = math.log1p(config.UNIT_MIX_MIN_LOG_SD)
    split = _otsu_split(x, w)
    parts = [(x[: split + 1], w[: split + 1]), (x[split + 1 :], w[split + 1 :])]
    pi = np.array([p[1].sum() for p in parts], dtype=float) / w.sum()
    mu = np.array([np.average(p[0], weights=p[1]) for p in parts])
    sd = np.array([max(math.sqrt(np.average((p[0] - m) ** 2, weights=p[1])), floor) for p, m in zip(parts, mu)])
    previous = likelihood = -math.inf
    for _ in range(config.UNIT_MIX_EM_ITERATIONS):
        density = np.vstack([pi[k] * _normal(x, mu[k], sd[k]) for k in range(2)]) + 1e-300
        total = density.sum(axis=0)
        likelihood = float((w * np.log(total)).sum())
        resp = density / total
        nk = (resp * w).sum(axis=1)
        if np.any(nk <= 0):
            break
        pi = nk / w.sum()
        mu = (resp * w * x).sum(axis=1) / nk
        sd = np.maximum(np.sqrt((resp * w * (x - mu[:, None]) ** 2).sum(axis=1) / nk), floor)
        if likelihood - previous < 1e-9:
            break
        previous = likelihood
    order = np.argsort(mu)
    return pi[order], mu[order], sd[order], likelihood


def _bic_gain(x: np.ndarray, w: np.ndarray, two_likelihood: float) -> float:
    """BIC of one normal minus BIC of the two-component fit (5 parameters against 2)."""
    floor = math.log1p(config.UNIT_MIX_MIN_LOG_SD)
    mu = float(np.average(x, weights=w))
    sd = max(math.sqrt(float(np.average((x - mu) ** 2, weights=w))), floor)
    one_likelihood = float((w * np.log(_normal(x, mu, sd) + 1e-300)).sum())
    return 2 * (two_likelihood - one_likelihood) - 3 * math.log(float(w.sum()))


def _round_between(low: float, high: float) -> float | None:
    """The roundest number strictly between ``low`` and ``high`` (35 between 27.5 and 41)."""
    centre = math.sqrt(low * high) if low > 0 else high / 2
    for digits in range(1, 13):
        candidate = float(f"{centre:.{digits}g}")
        if low < candidate < high:
            return candidate
    return None


def _share_inside(values: np.ndarray, weights: np.ndarray, bounds: config.UnitBounds) -> float:
    inside = (values >= bounds.minimum) & (values <= bounds.maximum)
    return float(weights[inside].sum() / weights.sum())


def detect_unit_mix(column: str, series: pl.Series, weights: pl.Series | None = None) -> UnitMix | None:
    """``column``'s values (``series``, or its distinct values with ``weights`` rows each) read as
    a mix of its unit pair's two units, or None (no pair, too few values, or one unit)."""
    pair = unit_pair_for(column)
    if pair is None:
        return None
    frame = numbers_with_counts(series, weights)
    minimum = config.UNIT_MIX_MIN_CLUSTER_VALUES
    if frame is None or int(frame.get_column("w").sum()) < 2 * minimum:
        return None
    frame = frame.sort("v")
    values = frame.get_column("v").to_numpy()
    counts = frame.get_column("w").to_numpy().astype(float)
    positive = values > 0
    if np.unique(values[positive]).size < 2:
        return None
    x, w = np.log(values[positive]), counts[positive]
    pi, mu, sd, likelihood = _mixture(x, w)
    separation = math.sqrt(2) * (mu[1] - mu[0]) / math.sqrt(sd[0] ** 2 + sd[1] ** 2)
    expected = math.exp(mu[0]) * pair.scale + pair.offset
    if separation <= config.UNIT_MIX_MIN_SEPARATION or expected <= 0:
        return None
    if _bic_gain(x, w, likelihood) < config.UNIT_MIX_MIN_BIC_GAIN:
        return None
    factor = math.log(expected) - mu[0]
    tolerance = min(math.log(config.UNIT_MIX_RATIO_TOLERANCE), config.UNIT_MIX_RATIO_TOLERANCE_SHARE * factor)
    if abs(math.log(expected) - mu[1]) > tolerance:
        return None
    # Hard assignment: the more likely component; values at or below zero belong to the low unit.
    logged = np.log(np.where(positive, values, 1.0))
    dens_low = _normal(logged, mu[0], sd[0]) * pi[0]
    dens_high = _normal(logged, mu[1], sd[1]) * pi[1]
    high = positive & (dens_high > dens_low)
    low = ~high
    low_count, high_count = int(counts[low].sum()), int(counts[high].sum())
    if counts[low & positive].sum() < minimum or high_count < minimum:
        return None
    share = config.UNIT_MIX_IN_RANGE_SHARE
    if _share_inside(values[low], counts[low], pair.low) < share:
        return None
    if _share_inside(values[high], counts[high], pair.high) < share:
        return None
    # Clean: every value is told apart with near certainty, and one cut point separates them.
    posterior = dens_high / (dens_low + dens_high + 1e-300)
    certain = config.UNIT_MIX_CLEAN_POSTERIOR
    ambiguous = positive & (posterior > 1 - certain) & (posterior < certain)
    top_low, bottom_high = float(values[low].max()), float(values[high].min())
    gap = math.log(bottom_high / top_low) if top_low > 0 and bottom_high > top_low else 0.0
    clean = not ambiguous.any() and gap >= config.UNIT_MIX_CLEAN_GAP_SHARE * factor
    cutoff = _round_between(top_low, bottom_high) if clean else None
    named = header_unit(column, pair)
    if named is None:
        named = pair.low.unit if low_count > high_count else pair.high.unit
    return UnitMix(column=column, pair=pair, low_count=low_count, high_count=high_count, cutoff=cutoff, target=named)
