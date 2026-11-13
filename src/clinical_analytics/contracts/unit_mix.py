# SPDX-License-Identifier: MIT
"""Find a column that mixes the two units of one variable (DQ M3).

Glucose recorded partly in mg/dL and partly in mmol/L has no meaningful mean ("Mean glucose was
35.61", an average of two scales). Only variables with a configured unit pair are checked
(``contracts.config.UNIT_PAIRS``). The whole column is read, before any limit or grouping of a
question, and it is read as mixed when any of these holds (``UnitMix.reason``):

* ``clusters``: a two-component normal mixture fitted to its log values separates cleanly
  (Ashman's D above ``UNIT_MIX_MIN_SEPARATION``), explains the values much better than one
  distribution (BIC gain of at least ``UNIT_MIX_MIN_BIC_GAIN``), and its centres differ by the
  unit factor (within ``UNIT_MIX_RATIO_TOLERANCE`` and ``UNIT_MIX_RATIO_TOLERANCE_SHARE``);
* ``factor``: the two units at their exact factor, one spread shared, explain the values better
  than one distribution (BIC gain above ``UNIT_MIX_MIN_FACTOR_GAIN``). This finds a small mix
  (30 kg beside 30 lb) the free fit misses, and a minority of one or two values set apart from
  the rest by an empty stretch (2 of 40 glucose values in mg/dL);
* ``implausible``: values implausible in the column's main unit but plausible in the other, set
  apart from the rest (98.6 in a °C column), whatever any fit says;
* ``spread``: no fit separates two groups, but the column is as wide as an even mix of the two
  units would be (``UNIT_MIX_SPREAD_SHARE``).

Each group must hold at least ``UNIT_MIX_MIN_CLUSTER_VALUES`` values (fewer when set apart), and
nearly all of it (``UNIT_MIX_IN_RANGE_SHARE``) must be plausible in its own unit. Two clean groups
at a ratio other than the unit factor (children beside adults at 3.9x) are not a mix.

Values are assigned to a group by one cut point between the two centres, never by which curve is
higher, so a wide group cannot claim values at both ends. The result holds counts and a cut
point only. No patient value leaves this module: the cut point is a round number strictly
between the groups, never a value from the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal, TypeAlias

import numpy as np
import polars as pl

from clinical_analytics.contracts import config
from clinical_analytics.contracts.builder import find_plausibility_range, numbers_with_counts, plausibility_key
from clinical_analytics.contracts.declarations import UnitRule

MixReason: TypeAlias = Literal["clusters", "factor", "implausible", "spread"]


def _pair_aliases(pair: config.UnitPair) -> frozenset[str]:
    found = next((r for r in config.PLAUSIBILITY_RANGES if r.variable == pair.variable), None)
    return pair.aliases | (found.aliases | found.hinted_aliases if found is not None else frozenset())


# Each pair's header aliases, read once: its own (weight, height) and its plausibility range's.
_ALIASES: Final[dict[str, frozenset[str]]] = {pair.variable: _pair_aliases(pair) for pair in config.UNIT_PAIRS}


@lru_cache(maxsize=4096)
def unit_pair_for(column: str) -> config.UnitPair | None:
    """The unit pair of ``column``: by its plausibility range (glucose, creatinine, temperature),
    else by the pair's aliases, a unit suffix allowed ("weight_kg", "temp_c", "Temp (°F)")."""
    found = find_plausibility_range(column)
    key = plausibility_key(column)
    for pair in config.UNIT_PAIRS:
        if (found is not None and found.variable == pair.variable) or _alias_of(key, pair) is not None:
            return pair
    return None


def _alias_of(key: str, pair: config.UnitPair) -> str | None:
    """The unit a header names after an alias ("glucosemgdl" -> "mg/dL", "tempf" -> "°F"), ""
    for a bare alias, None when ``key`` is not an alias of the pair."""
    aliases = _ALIASES[pair.variable]
    for unit, hints in ((pair.low.unit, pair.low_hints), (pair.high.unit, pair.high_hints)):
        if any(key.endswith(hint) and key[: -len(hint)] in aliases for hint in hints):
            return unit
    return "" if key in aliases else None


def header_unit(column: str, pair: config.UnitPair) -> str | None:
    """The unit ``column``'s header names ("Glucose (mg/dL)" -> "mg/dL"), or None. Evidence of
    which unit is meant, never proof that the column holds only that unit."""
    return _alias_of(plausibility_key(column), pair) or None


@dataclass(frozen=True)
class UnitMix:
    """A column read as two groups, one per unit of ``pair``.

    ``low_count``/``high_count``: rows read as the low/high unit (every number the check read is
    in one of them, values at or below zero in the low one). ``cutoff``: a round number strictly
    between the groups, or None when they overlap (the empty stretch between them is narrower
    than ``UNIT_MIX_CLEAN_GAP_SHARE`` of the units' log distance). ``target``: the unit the
    header names, else the larger group's (the high unit on a tie). ``one_units``: the units the
    doctor may declare every value to be in (children beside adults, all in kg): the header's
    unit when it names one, else each unit, ``target`` first; only a unit every value is
    plausible in. ``reason``: which test found the mix (module docstring).
    """

    column: str
    pair: config.UnitPair
    low_count: int
    high_count: int
    cutoff: float | None
    target: str
    reason: MixReason = "clusters"
    one_units: tuple[str, ...] = ()

    @property
    def present(self) -> int:
        return self.low_count + self.high_count

    @property
    def other(self) -> str:
        return self.pair.low.unit if self.target == self.pair.high.unit else self.pair.high.unit

    def rule(self, dataset_version: str | None) -> UnitRule | None:
        """The conversion the doctor may confirm: values on the other unit's side of the cut
        point are read in that unit and converted to ``target``. None when the groups overlap
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


def _floor() -> float:
    return math.log1p(config.UNIT_MIX_MIN_LOG_SD)


def _log_factor(pair: config.UnitPair, log_low: float) -> float:
    """Log distance from a low-unit value to the same quantity in the high unit (constant for a
    pure factor; for °C/°F it depends on the value)."""
    return math.log(math.exp(log_low) * pair.scale + pair.offset) - log_low


def _otsu_split(x: np.ndarray, w: np.ndarray) -> int:
    """Index of the last low value of sorted ``x`` that maximises the between-cluster variance."""
    cw = np.cumsum(w)[:-1]
    cs = np.cumsum(w * x)[:-1]
    total_w, total_s = float(w.sum()), float((w * x).sum())
    m0, m1 = cs / cw, (total_s - cs) / (total_w - cw)
    return int(np.argmax(cw * (total_w - cw) * (m0 - m1) ** 2))


def _one_normal(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """(mean, spread, log-likelihood) of one normal fitted to ``x`` (counts ``w``)."""
    mu = float(np.average(x, weights=w))
    sd = max(math.sqrt(float(np.average((x - mu) ** 2, weights=w))), _floor())
    return mu, sd, float((w * np.log(_normal(x, mu, sd) + 1e-300)).sum())


def _mixture(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(weights, means, spreads, log-likelihood) of a two-component normal mixture fitted to
    sorted ``x`` (counts ``w``) by EM, started from the Otsu split; component 0 has the lower
    mean."""
    floor = _floor()
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


def _factor_fit(x: np.ndarray, w: np.ndarray, pair: config.UnitPair) -> tuple[float, float]:
    """(BIC gain over one normal, low centre) of two components exactly one unit factor apart
    with one shared spread (3 parameters against 2), fitted by EM from several starting shares."""
    floor, total = _floor(), float(w.sum())
    mean, spread, one_likelihood = _one_normal(x, w)
    best, best_low = -math.inf, mean
    for start in (0.05, 0.25, 0.5, 0.75, 0.95):
        share, low, sd = start, mean - start * _log_factor(pair, mean), spread
        likelihood = previous = -math.inf
        for _ in range(config.UNIT_MIX_EM_ITERATIONS):
            factor = _log_factor(pair, low)
            density_low = (1 - share) * _normal(x, low, sd) + 1e-300
            density_high = share * _normal(x, low + factor, sd) + 1e-300
            likelihood = float((w * np.log(density_low + density_high)).sum())
            high = density_high / (density_low + density_high)
            share = float((w * high).sum() / total)
            if not 0 < share < 1 or likelihood - previous < 1e-9:
                break
            low = float((w * ((1 - high) * x + high * (x - factor))).sum() / total)
            sd = max(
                math.sqrt(float((w * ((1 - high) * (x - low) ** 2 + high * (x - low - factor) ** 2)).sum() / total)),
                floor,
            )
            previous = likelihood
        if likelihood > best:
            best, best_low = likelihood, low
    return 2 * (best - one_likelihood) - math.log(total), best_low


def _round_between(low: float, high: float) -> float | None:
    """The roundest number strictly between ``low`` and ``high`` (35 between 27.5 and 41)."""
    centre = math.sqrt(low * high) if low > 0 else high / 2
    for digits in range(1, 13):
        candidate = float(f"{centre:.{digits}g}")
        if low < candidate < high:
            return candidate
    return None


def _inside(values: np.ndarray, bounds: config.UnitBounds) -> np.ndarray:
    return (values >= bounds.minimum) & (values <= bounds.maximum)


def _share_inside(values: np.ndarray, weights: np.ndarray, bounds: config.UnitBounds) -> float:
    return float(weights[_inside(values, bounds)].sum() / weights.sum()) if weights.sum() else 0.0


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Median of sorted ``x`` with counts ``w``."""
    return float(x[int(np.searchsorted(np.cumsum(w), w.sum() / 2))])


@dataclass(frozen=True)
class _Split:
    """A mix found by one test: the log cut point between the groups and the log unit distance."""

    reason: MixReason
    cut: float
    factor: float


def _groups_hold(v: np.ndarray, w: np.ndarray, split: _Split, pair: config.UnitPair, few: bool) -> bool:
    """Each side of ``split`` is plausible in its unit and holds enough values: at least
    ``UNIT_MIX_MIN_CLUSTER_VALUES``, or (``few``) one when an empty stretch sets it apart."""
    high = np.log(v) > split.cut
    if not high.any() or high.all():
        return False
    share = config.UNIT_MIX_IN_RANGE_SHARE
    if _share_inside(v[~high], w[~high], pair.low) < share or _share_inside(v[high], w[high], pair.high) < share:
        return False
    minimum = config.UNIT_MIX_MIN_CLUSTER_VALUES
    smaller = min(float(w[~high].sum()), float(w[high].sum()))
    if smaller >= minimum:
        return True
    bulk = max(float(w[~high].sum()), float(w[high].sum()))
    return few and bulk >= minimum and _gap(v, high) >= config.UNIT_MIX_CLEAN_GAP_SHARE * split.factor


def _gap(v: np.ndarray, high: np.ndarray) -> float:
    """Log width of the empty stretch between the low group's top and the high group's bottom."""
    top_low, bottom_high = float(v[~high].max()), float(v[high].min())
    return math.log(bottom_high / top_low) if bottom_high > top_low else 0.0


def _find_split(v: np.ndarray, w: np.ndarray, pair: config.UnitPair, header: str | None) -> _Split | None:
    """The first test that reads positive values ``v`` (sorted, counts ``w``) as a mix."""
    x = np.log(v)
    pi, mu, sd, likelihood = _mixture(x, w)
    separation = math.sqrt(2) * (mu[1] - mu[0]) / math.sqrt(sd[0] ** 2 + sd[1] ** 2)
    factor = _log_factor(pair, float(mu[0]))
    tolerance = min(math.log(config.UNIT_MIX_RATIO_TOLERANCE), config.UNIT_MIX_RATIO_TOLERANCE_SHARE * factor)
    at_factor = abs(float(mu[0]) + factor - float(mu[1])) <= tolerance
    separated = separation > config.UNIT_MIX_MIN_SEPARATION
    gain = 2 * (likelihood - _one_normal(x, w)[2]) - 3 * math.log(float(w.sum()))
    if separated and at_factor and gain >= config.UNIT_MIX_MIN_BIC_GAIN:
        # Between the centres, each at the same number of its own spreads.
        cut = float((mu[0] * sd[1] + mu[1] * sd[0]) / (sd[0] + sd[1]))
        split = _Split("clusters", cut, factor)
        if _groups_hold(v, w, split, pair, few=False):
            return split
    found = _implausible_split(v, w, pair, header)
    if found is not None:
        return found
    factor_gain, low = _factor_fit(x, w, pair)
    if separated and not at_factor and gain > factor_gain:
        return None  # two clear groups better read at another ratio: children beside adults
    factor = _log_factor(pair, low)
    if factor_gain > config.UNIT_MIX_MIN_FACTOR_GAIN:
        split = _Split("factor", low + factor / 2, factor)
        if _groups_hold(v, w, split, pair, few=True):
            return split
    spread_factor = _log_factor(pair, _weighted_median(x, w))
    if _one_normal(x, w)[1] >= config.UNIT_MIX_SPREAD_SHARE * spread_factor:
        split = _Split("spread", low + factor / 2, spread_factor)
        if _groups_hold(v, w, split, pair, few=True):
            return split
    return None


def _implausible_split(v: np.ndarray, w: np.ndarray, pair: config.UnitPair, header: str | None) -> _Split | None:
    """Values implausible in the column's main unit but plausible in the other, set apart from
    the rest. The main unit is the header's, else the only unit the median is plausible in (a
    median of 110 is plausible as mg/dL and as mmol/L, so no main unit is known)."""
    median = _weighted_median(v, w)
    plausible = [u for u in (pair.low, pair.high) if u.minimum <= median <= u.maximum]
    for main in (pair.low, pair.high):
        if (header is not None and header != main.unit) or (header is None and plausible != [main]):
            continue
        other = pair.high if main is pair.low else pair.low
        inside = _inside(v, main)
        strays = ~inside & _inside(v, other) & ((v > main.maximum) if main is pair.low else (v < main.minimum))
        if not strays.any() or not inside.any():
            continue
        if main is pair.low:
            edge_low, edge_high = float(v[inside].max()), float(v[strays].min())
        else:
            edge_low, edge_high = float(v[strays].max()), float(v[inside].min())
        factor = _log_factor(pair, math.log(median) if main is pair.low else math.log(edge_low))
        if edge_high > edge_low and math.log(edge_high / edge_low) >= config.UNIT_MIX_CLEAN_GAP_SHARE * factor:
            return _Split("implausible", math.log(math.sqrt(edge_low * edge_high)), factor)
    return None


def detect_unit_mix(
    column: str, series: pl.Series, weights: pl.Series | None = None, pair: config.UnitPair | None = None
) -> UnitMix | None:
    """``column``'s values (``series``, or its distinct values with ``weights`` rows each) read as
    a mix of its unit pair's two units, or None (no pair, too few values, or one unit).
    ``pair``: the column's unit pair when the caller has already looked it up."""
    pair = pair if pair is not None else unit_pair_for(column)
    if pair is None:
        return None
    frame = numbers_with_counts(series, weights)
    if frame is None or frame.height == 0:
        return None
    frame = frame.sort("v")
    values = frame.get_column("v").to_numpy()
    counts = frame.get_column("w").to_numpy().astype(float)
    positive = values > 0
    if np.unique(values[positive]).size < 2:
        return None
    header = header_unit(column, pair)
    v, w = values[positive], counts[positive]
    split = _find_split(v, w, pair, header)
    if split is None:
        return None
    high = positive & (np.log(np.where(positive, values, 1.0)) > split.cut)
    low = ~high
    low_count, high_count = int(counts[low].sum()), int(counts[high].sum())
    clean = _gap(v, high[positive]) >= config.UNIT_MIX_CLEAN_GAP_SHARE * split.factor
    top_low, bottom_high = float(v[~high[positive]].max()), float(v[high[positive]].min())
    cutoff = _round_between(top_low, bottom_high) if clean else None
    target = header if header is not None else (pair.low.unit if low_count > high_count else pair.high.unit)
    other = pair.high if target == pair.low.unit else pair.low
    candidates = [pair.low if target == pair.low.unit else pair.high, *([] if header is not None else [other])]
    one_units = tuple(bounds.unit for bounds in candidates if bool(_inside(v, bounds).all()))
    return UnitMix(
        column=column,
        pair=pair,
        low_count=low_count,
        high_count=high_count,
        cutoff=cutoff,
        target=target,
        reason=split.reason,
        one_units=one_units,
    )
