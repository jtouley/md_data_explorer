# SPDX-License-Identifier: MIT
"""Aggregate-data mode: questions about files whose rows are estimates, not patients (ADR014).

A population-estimate file (CDC tables, WHO TB, GBD extracts) has one row per combination of
its key dimensions (location, year, sex, age, cause, measure) and a value that often comes with
lower and upper bounds. Treating its rows as patients gives wrong answers: a mean of rates across
"Male", "Female" and "Both", across a state and its own counties and across years, labelled as
n patients; an ANOVA across the three sex rows. This module:

* detects such a table (``detect_estimate_table``): the dimensions that key a row, the value
  columns and their bounds, and the levels that are totals of the others ("Both", "All ages",
  "United States", a state beside its counties, an age band that contains other bands);
* checks a parsed plan against it (``check_plan``): a dimension the question leaves open is never
  averaged over. The question is held when every open dimension has a total level to offer
  (confirming saves that choice for the dataset) and refused, naming what to fix, otherwise;
* answers by reading the estimate rows (``estimate_answer``): each value with its interval, a
  total level left out of a comparison of its parts, and "different" only when the intervals do
  not overlap. No statistical test runs on estimates.

Detection alone never relaxes a privacy rule: answering needs the dataset to be declared public
population estimates at upload (``Declarations.population_estimates``). Without that the rows are
treated like patients, and every group of one row is a small cell.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

import polars as pl
import structlog

from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts import config
from clinical_analytics.core.identifier_rules import ROW_NUMBER_COLUMN
from clinical_analytics.core.query_plan import FilterSpec, QueryPlan
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED

logger = structlog.get_logger()

ValueKind: TypeAlias = Literal["rate", "count", "value"]
# Plan intents whose statistic would treat the rows as patients (a model, a survival curve, a
# correlation across rows): refused on estimates.
_PATIENT_TESTS: Final = frozenset({"FIND_PREDICTORS", "CORRELATIONS", "SURVIVAL"})
_STATISTIC_INTENTS: Final = frozenset({"DESCRIBE", "COMPARE_GROUPS"})
_LOWEST_WORDS: Final = re.compile(r"\b(lowest|least|smallest|bottom|minimum|fewest)\b", re.IGNORECASE)
_AGE_BAND = re.compile(
    r"^\s*(?:(?P<lo>\d+)\s*(?:-|–|to)\s*(?P<hi>\d+)|(?P<plus>\d+)\s*(?:\+|plus|years? and (?:over|older))"
    r"|under\s*(?P<under>\d+))",
    re.IGNORECASE,
)
_OPEN_AGE_TOP: Final = 200
# Columns the semantic layer adds for patient cohorts; skipped when detecting an estimate table.
APP_ADDED_COLUMNS: Final = ("time_zero", "outcome_label", "features_json", ROW_NUMBER_COLUMN)
_CI_STEM: Final = re.compile(r"(^|_)(ci|confidence|interval)($|_)")


class EstimateGrainError(ValueError):
    """Two rows answer one group: the estimate key did not pin every row (fail closed)."""


@dataclass(frozen=True)
class ValueColumn:
    """A numeric column of estimates, with its interval bounds when the file has them."""

    name: str
    lower: str | None
    upper: str | None
    kind: ValueKind
    # Flag columns staging split out of the source's markers ("Suppressed", "(Unreliable)").
    suppressed: str | None = None
    unreliable: str | None = None


@dataclass(frozen=True)
class Dimension:
    """A column that keys the estimate rows, its level count, and the levels that are totals of
    other levels (``top``: the one total of every other level, when there is one)."""

    name: str
    levels: int
    parents: tuple[Any, ...] = ()
    top: Any = None
    yearlike: bool = False
    aliases: tuple[str, ...] = ()
    values: tuple[Any, ...] = ()  # its levels (published category labels), when few enough to list


@dataclass(frozen=True)
class EstimateTable:
    """What makes the rows estimates: the key dimensions, the value columns, the other columns."""

    rows: int
    dimensions: tuple[Dimension, ...]
    values: tuple[ValueColumn, ...]
    attributes: tuple[str, ...] = ()
    # Metric-like columns (a key, an attribute or a one-value column) -> level -> what that level
    # makes a row's value: "rate" (Rate, Percent, Prevalence) or "count" (Number, anything else).
    metric_levels: Mapping[str, Mapping[Any, str]] = field(default_factory=dict)
    # Location dimension -> county-like level -> the level of its own state in that dimension, from
    # a five-digit FIPS alias or a ", CO"/", Colorado" suffix: counties nest under their own state.
    nesting: Mapping[str, Mapping[Any, Any]] = field(default_factory=dict)

    @property
    def count_levels(self) -> dict[str, tuple[Any, ...]]:
        """Metric column -> its levels that make a row a count."""
        found = {
            c: tuple(lv for lv, kind in levels.items() if kind == "count") for c, levels in self.metric_levels.items()
        }
        return {c: lv for c, lv in found.items() if lv}

    def dimension(self, column: str | None) -> Dimension | None:
        for dimension in self.dimensions:
            if column == dimension.name or column in dimension.aliases:
                return dimension
        return None

    def value(self, column: str | None) -> ValueColumn | None:
        for value in self.values:
            if column in (value.name, value.lower, value.upper):
                return value
        return None

    def primary_value(self) -> ValueColumn | None:
        """The one value column a question that names none can mean, or None when there are several."""
        bounded = [value for value in self.values if value.lower and value.upper]
        candidates = bounded or list(self.values)
        return candidates[0] if len(candidates) == 1 else None

    def describe(self) -> str:
        """'one row per location_name × year_id × sex' (the key, in the file's own names)."""
        return "one row per " + " × ".join(d.name for d in self.dimensions if d.levels > 1)


@dataclass
class EstimateCheck:
    """What ``check_plan`` decided: refuse, hold (with the totals confirming would save), or run."""

    refusal: str | None = None
    hold: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    value: ValueColumn | None = None
    group_dims: tuple[str, ...] = ()
    pinned: dict[str, Any] = field(default_factory=dict)
    left_out: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    chosen: dict[str, Any] = field(default_factory=dict)  # saved choices this answer applies
    frame: pl.DataFrame | None = None


# ---------------------------------------------------------------------------- detection


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def _bound_pairs(columns: Sequence[str], numeric: set[str]) -> dict[str, tuple[str, str]]:
    """value column -> (lower, upper), matched by name (``config.ESTIMATE_BOUND_SUFFIXES`` and the
    standalone pairs such as ``lower``/``upper`` beside one ``val``/``mx``/``datavalue``)."""
    by_norm = {_norm(column): column for column in columns if column in numeric}
    pairs: dict[str, tuple[str, str]] = {}
    for low_suffix, high_suffix in config.ESTIMATE_BOUND_SUFFIXES:
        for norm, column in by_norm.items():
            if not norm.endswith(low_suffix):
                continue
            stem = norm[: -len(low_suffix)]
            upper = by_norm.get(stem + high_suffix)
            value = by_norm.get(stem)
            if upper is not None and value is not None and value not in pairs:
                pairs[value] = (column, upper)
    for low, high in config.ESTIMATE_STANDALONE_BOUNDS:
        if low in by_norm and high in by_norm:
            used = {c for pair in pairs.values() for c in pair}
            values = [
                by_norm[name]
                for name in config.ESTIMATE_VALUE_NAMES
                if name in by_norm and by_norm[name] not in pairs and by_norm[low] not in used
            ]
            if len(values) == 1:
                pairs[values[0]] = (by_norm[low], by_norm[high])
    # "95% CI (%) lower" / "... upper" split from one interval column (FluVaxView), beside one value.
    for norm, column in by_norm.items():
        ci_stem = norm[: -len("_lower")] if norm.endswith("_lower") else None
        if ci_stem is None or not _CI_STEM.search(ci_stem) or ci_stem + "_upper" not in by_norm:
            continue
        used = {c for pair in pairs.values() for c in pair}
        values = [
            by_norm[name] for name in config.ESTIMATE_VALUE_NAMES if name in by_norm and by_norm[name] not in pairs
        ]
        if len(values) == 1 and column not in used:
            pairs[values[0]] = (column, by_norm[ci_stem + "_upper"])
    _labelled_bound_pairs(by_norm, pairs)
    return pairs


# "<value> Lower 95% Confidence Interval", "Lower 95% CI", "lci": a side word, an optional level
# and an optional interval word, after the value's own name (or alone, beside one rate).
_SIDE_BOUND: Final = re.compile(
    r"^(?P<stem>.*?)_?(?P<side>lower|upper|low|high|lci|uci|lcl|ucl)(?P<tail>(_\d+)?(_(pct|percent))?"
    r"(_(ci|cl|confidence_interval|confidence_limit|interval|limit|bound|confidence))?)$"
)
_LOWER_SIDES: Final = {"lower": "upper", "low": "high", "lci": "uci", "lcl": "ucl"}


def _labelled_bound_pairs(by_norm: Mapping[str, str], pairs: dict[str, tuple[str, str]]) -> None:
    """Pairs the earlier passes missed: WONDER's "Crude Rate Lower 95% Confidence Interval" with
    "Crude Rate", and AtlasPlus's standalone "Lower 95% CI"/"Upper 95% CI" with its one rate."""
    used = {c for pair in pairs.values() for c in pair}
    for norm, column in by_norm.items():
        match = _SIDE_BOUND.match(norm)
        if match is None or column in used or match.group("side") not in _LOWER_SIDES:
            continue
        stem, tail = match.group("stem"), match.group("tail")
        if not stem and not tail and match.group("side") in ("low", "lower"):
            continue  # a bare "lower"/"low" is the standalone pass's
        high = _LOWER_SIDES[match.group("side")]
        upper = by_norm.get(f"{stem}_{high}{tail}" if stem else f"{high}{tail}")
        if upper is None or upper in used:
            continue
        if stem:
            value = by_norm.get(stem)
            candidates = [value] if value is not None else []
        else:
            candidates = [
                c
                for n, c in by_norm.items()
                if c not in used
                and c not in (column, upper)
                and _SIDE_BOUND.match(n) is None
                and value_kind(c) == "rate"
            ]
        if len(candidates) == 1 and candidates[0] not in pairs:
            pairs[candidates[0]] = (column, upper)
            used |= {column, upper}


def _flag(column: str, suffix: str, flags: set[str]) -> str | None:
    name = f"{column}{suffix}"
    return name if name in flags else None


def value_kind(column: str) -> ValueKind:
    """Rate or count, from the column name, for comparing tables (a rate is never compared with a
    count). Only a name that reads as a rate, ratio or percent ("Rate per 100000", "Age-adjusted
    Death Rate", "e_inc_100k", "Estimate (%)") is a rate; a metric level on the row can say so too
    (``estimate_join.measure_of``). Kind decides comparisons only: published values are shown as
    published (owner's decision, ADR014 §7)."""
    if config.ESTIMATE_RATE_NAME.search(_norm(column)) or "%" in column:
        return "rate"
    return "count"


def metric_level_kind(level: Any) -> Literal["rate", "count"]:
    """What a metric level says its row's value is: a rate ("Rate", "Percent", "Crude Prevalence",
    "Age-adjusted Rate") or, for anything else ("Number", "Mean", an unknown label), a count."""
    if "%" in str(level):
        return "rate"  # "%" is a percent
    norm = _norm(str(level))
    if norm in config.ESTIMATE_COUNT_LEVELS:
        return "count"
    return "rate" if config.ESTIMATE_RATE_LEVEL.search(norm) else "count"


def _age_band(level: Any) -> tuple[int, int] | None:
    if not isinstance(level, str):
        return None
    match = _AGE_BAND.match(level)
    if match is None:
        return None
    if match.group("lo") is not None:
        return int(match.group("lo")), int(match.group("hi"))
    if match.group("plus") is not None:
        return int(match.group("plus")), _OPEN_AGE_TOP
    return 0, max(int(match.group("under")) - 1, 0)


def parent_levels(levels: Sequence[Any], column: str = "") -> tuple[tuple[Any, ...], Any]:
    """(levels that are totals of other levels, the total of all of them or None).

    A total is a level named as one ("Both", "All ages", "United States", "Overall"), an age band
    that contains another band present ("65+" beside "65-74") in a column named for age, or, in a
    column where some levels are counties, every level that is not ("Colorado" beside "Baca County").
    """
    texts = [level for level in levels if isinstance(level, str)]
    if len(levels) < 2:
        return (), None
    if None in levels and len(levels) == 2:
        return (None,), None
    named = [level for level in texts if level.strip().casefold() in config.ESTIMATE_TOTAL_LEVELS]
    # "Age-standardized" summarises every age as "All ages" does, by another method: both are
    # summary levels, and with both present neither is the one total (so no default is offered).
    standardized = [level for level in texts if _is_standardized_level(level)]
    named += standardized
    folded = {level.strip().casefold() for level in texts}
    if {"m", "f", "a"} <= folded:
        named += [level for level in texts if level.strip().casefold() == "a"]
    is_age = "age" in _norm(column).split("_")
    bands = {level: band for level in texts if is_age and (band := _age_band(level)) is not None}
    containing = [
        level
        for level, (lo, hi) in bands.items()
        if any(other != level and lo <= olo and ohi <= hi for other, (olo, ohi) in bands.items())
    ]
    counties = [level for level in texts if config.ESTIMATE_SUBAREA_NAME.search(level)]
    # A non-county level beside counties is their total (the state row), unless it is itself part
    # of a state: Detroit and out-Wayne are local jurisdictions, never a total of the counties.
    areas = [
        level
        for level in texts
        if counties and level not in counties and level not in named and _place_kind(level, column) != "substate"
    ]
    # An empty level beside real ones is a total row (CDC WONDER's "Total" rows leave Gender empty).
    empty = [None] if None in levels else []
    parents = tuple(dict.fromkeys([*named, *containing, *areas, *empty]))
    top: Any = None
    if len(named) == 1:
        top = named[0]
    elif not named and len(areas) == 1:
        top = areas[0]
    elif not named and containing and len(bands) == len(texts):
        widest = [
            level
            for level in containing
            if all(bands[level][0] <= lo and hi <= bands[level][1] for lo, hi in bands.values())
        ]
        top = widest[0] if len(widest) == 1 else None
    return parents, top


def _is_standardized_level(level: Any) -> bool:
    return isinstance(level, str) and _norm(level) in config.ESTIMATE_STANDARDIZED_LEVELS


def is_metric_column(column: str) -> bool:
    """A column whose levels say what the value is (GBD ``metric_name``, CDC ``DataValueType``)."""
    return bool(config.ESTIMATE_METRIC_DIMENSION.search(_norm(column)))


def _county_states(view: Any, dimension: Dimension) -> dict[Any, Any]:
    """County-like level -> its state's level in the same dimension (empty when unknown)."""
    from clinical_analytics.contracts.geography import classify, is_geo_code_column

    states = {}
    for level in dimension.values:
        place = classify(level)
        if place.kind == "state" and place.state is not None:
            states[place.state.fips] = level
    counties = [
        v for v in dimension.values if isinstance(v, str) and config.ESTIMATE_SUBAREA_NAME.search(v.split(",")[0])
    ]
    if not states or not counties:
        return {}
    found: dict[Any, Any] = {}
    for alias in dimension.aliases:
        if not is_geo_code_column(alias):
            continue
        pairs = view.select(dimension.name, alias).distinct().to_polars()
        for level, code in pairs.iter_rows():
            digits = str(code).strip()
            if level in counties and digits.isdigit() and len(digits) in (4, 5):
                state = states.get(digits.zfill(5)[:2])
                if state is not None:
                    found[level] = state
    for level in counties:
        if level in found or "," not in level:
            continue
        suffix = classify(level.rsplit(",", 1)[1].strip())
        if suffix.kind == "state" and suffix.state is not None and suffix.state.fips in states:
            found[level] = states[suffix.state.fips]
    return found


def _place_kind(level: Any, column: str) -> str:
    from clinical_analytics.contracts.geography import classify, is_geo_code_column

    return classify(level, is_geo_code_column(column)).kind


def _mostly_places(levels: Sequence[Any], column: str) -> bool:
    present = [level for level in levels if level is not None]
    return len(present) >= 2 and sum(_place_kind(level, column) != "unknown" for level in present) * 2 >= len(present)


def _mostly_states(levels: Sequence[Any], column: str) -> bool:
    present = [level for level in levels if level is not None]
    # States, territories and regions (census or HHS): one row each is an aggregate table.
    kinds = ("state", "territory", "region")
    return len(present) >= 2 and sum(_place_kind(level, column) in kinds for level in present) * 2 >= len(present)


def _is_dimension_name(column: str) -> bool:
    return bool(config.ESTIMATE_KEY_NAME.search(_norm(column)))


def _is_year(column: str, levels: Sequence[Any]) -> bool:
    if re.search(r"year|season", _norm(column)):
        return True
    numbers = [level for level in levels if isinstance(level, int) and not isinstance(level, bool)]
    return (
        bool(numbers)
        and len(numbers) == len([v for v in levels if v is not None])
        and all(1800 <= n <= 2200 for n in numbers)
    )


def _public_shape_problem(
    view: Any,
    schema: Any,
    key: Sequence[str],
    listed_by: Mapping[str, list[Any]],
    levels: Mapping[str, int],
    candidates: Sequence[str],
    aliases: Mapping[str, list[str]],
    numeric: set[str],
    pairs: Mapping[str, tuple[str, str]],
    values: Sequence[ValueColumn],
    metric_levels: Mapping[str, Mapping[Any, str]],
) -> str | None:
    """Which allowlist rule a table breaks (``core/public_shape.py``), or None when it is shaped like
    a public estimate table. Fail closed: a key too large to list is never recognised."""
    from clinical_analytics.core import public_shape

    def distinct(column: str) -> list[Any]:
        if levels.get(column, 0) > config.ESTIMATE_MAX_LISTED_LEVELS:
            return []
        found: list[Any] = view.select(column).distinct().to_polars().get_column(column).to_list()
        return found

    # A column named as a person or record ID refuses the table whatever its values, key or not.
    if any(public_shape.person_id_named(c) for c in (*candidates, *numeric)):
        return "person_id_column"
    kinds_seen: list[frozenset[str]] = []
    has_label = False
    for column in key:
        verdict = public_shape.key_column(column, listed_by[column], {a: distinct(a) for a in aliases.get(column, [])})
        if not verdict.ok:
            return "key_not_a_public_dimension"
        kinds_seen.append(verdict.kinds)
        has_label = has_label or verdict.label
    time_only = [
        c
        for c, kinds in zip(key, kinds_seen, strict=True)
        if "time" in kinds and not kinds - {"time", "total", "placeholder"}
    ]
    if len(time_only) > 1 and not public_shape.year_with_period(time_only, listed_by):
        return "two_time_keys"  # a year beside a visit month: an ID shaped like years, not a public grid
    all_kinds = frozenset().union(*kinds_seen) if kinds_seen else frozenset()
    # Every public source has a place: a key of places, or one place for the whole file (a state's
    # own table). A time-only or label-only table is refused (ADR014 §7).
    single_place = any(
        levels.get(c) == 1 and public_shape.classify_value(distinct(c)[0], c) == "place"
        for c in candidates
        if c not in key and schema[c].is_string()
    )
    if "place" not in all_kinds and not single_place:
        return "no_place"
    if has_label:
        rated = any(v.kind == "rate" for v in values) or any(
            kind == "rate" for levels_ in metric_levels.values() for kind in levels_.values()
        )
        if not (all_kinds & {"place", "time"} and (pairs or rated or "total" in all_kinds)):
            return "label_key_without_place_time_and_estimate"
    if not all_kinds & {"place", "time"} and not pairs and "total" not in all_kinds:
        # Demographics only (sex x age band x race), no interval and no total row: a patient file
        # with one row per person in each cell is shaped the same, so it is not opened here.
        return "demographics_only_without_interval_or_total"
    alias_of = {a: k for k, names in aliases.items() for a in names}
    for column in candidates:
        if column in key:
            continue
        # A code column for an accepted key column (GBD location_id, CDC stratificationid1): an alias
        # of one, a one-value column, or fixed by one key column. A per-row MRN is none of these.
        id_alias = public_shape.id_named(column) and (
            column in alias_of
            or levels.get(column) == 1
            or any(_distinct_rows(view, [k, column]) == _distinct_rows(view, [k]) for k in key)
        )
        if public_shape.other_column_problem(column, distinct(column), is_numeric=False, id_alias=id_alias):
            return "other_column_reveals_people"
    for column in numeric:
        if column not in candidates and public_shape.other_column_problem(column, [], is_numeric=True, id_alias=False):
            return "numeric_patient_id"
    return None


def _distinct_rows(view: Any, columns: Sequence[str]) -> int:
    return int(view.select(*columns).distinct().count().execute())


def detect_estimate_table(view: Any, rows: int, skip: Sequence[str] = ()) -> EstimateTable | None:
    """The estimate structure of ``view``, or None when its rows are not estimates.

    Rows are estimates when the text and code columns (minus columns that only repeat another,
    such as ``sex_id`` beside ``sex``) key every row exactly, no key column comes close to one
    value per row (that would be a person or record ID), and at least one numeric column is an
    estimate: it has lower and upper bounds, or its name says rate, ratio or percent.

    Every count is a DuckDB aggregate; only distinct levels of the key columns (at most
    ``config.ESTIMATE_MAX_LISTED_LEVELS`` each) are read into Python.
    """
    if rows < 2:
        return None
    schema = view.schema()
    columns = [c for c in view.columns if c not in skip and c not in APP_ADDED_COLUMNS]
    numeric = {c for c in columns if schema[c].is_numeric() and not schema[c].is_boolean()}
    pairs = _bound_pairs(columns, numeric)
    bound_columns = {c for pair in pairs.values() for c in pair}
    candidates: list[str] = []
    value_names: list[str] = []
    for column in columns:
        dtype = schema[column]
        if column in bound_columns:
            continue
        if dtype.is_string() or (dtype.is_integer() and _is_dimension_name(column)):
            candidates.append(column)
        elif column in numeric:
            value_names.append(column)
    flags = {c for c in columns if schema[c].is_boolean()}
    values = tuple(
        ValueColumn(
            name,
            pairs[name][0] if name in pairs else None,
            pairs[name][1] if name in pairs else None,
            value_kind(name),
            suppressed=_flag(name, config.SUPPRESSED_FLAG_SUFFIX, flags),
            unreliable=_flag(name, config.UNRELIABLE_FLAG_SUFFIX, flags),
        )
        for name in value_names
    )
    if not candidates or not values:
        return None
    metric_levels = {
        column: {
            level: metric_level_kind(level)
            for level in view.select(column).distinct().to_polars().get_column(column).drop_nulls().to_list()
        }
        for column in candidates
        # The named metric only (metric_name, DataValueType): a numeric code beside it (metric_id
        # 1/2/3) says nothing a reader can check, so it never decides a value's kind.
        if is_metric_column(column) and schema[column].is_string()
    }
    counts = view.aggregate(**{f"c{i}": view[c].nunique() for i, c in enumerate(candidates)}).to_polars().row(0)
    levels = {column: int(count) for column, count in zip(candidates, counts, strict=True)}
    candidates = [c for c in candidates if levels[c] > 0]
    if not candidates:
        return None  # every key-like column is empty: nothing keys the rows
    if _distinct_rows(view, candidates) != rows:
        return None
    # A column with the same levels as another one, row for row (sex_id beside sex), is an alias.
    aliases: dict[str, list[str]] = {}
    kept: list[str] = []
    # Distinct levels counted with the empty value, so a total row empty in both (WONDER's "Total"
    # rows: Gender and Gender Code both empty) doesn't stop two columns being aliases.
    with_null = {c: _distinct_rows(view, [c]) for c in candidates}
    for column in sorted(candidates, key=lambda c: (not schema[c].is_string(), candidates.index(c))):
        twin = next(
            (
                k
                for k in kept
                if with_null[k] == with_null[column] and _distinct_rows(view, [k, column]) == with_null[k]
            ),
            None,
        )
        if twin is None:
            kept.append(column)
        else:
            aliases.setdefault(twin, []).append(column)
    # Drop columns the others already determine (WHO region beside country): attributes, not keys.
    # Most levels first, so a record ID added at upload goes before a real key column does.
    key = list(kept)
    for column in sorted(kept, key=lambda c: -levels[c]):
        rest = [c for c in key if c != column]
        if rest and _distinct_rows(view, rest) == rows:
            key.remove(column)
    listed_by: dict[str, list[Any]] = {
        column: view.select(column).distinct().to_polars().get_column(column).to_list()
        if levels[column] <= config.ESTIMATE_MAX_LISTED_LEVELS
        else []
        for column in key
    }
    # One value per row (or nearly) is a record ID, so rows are records. A state column is the
    # exception: it has at most 51 levels, and one row per state is an aggregate table.
    if any(levels[c] * 2 > rows and not _mostly_states(listed_by[c], c) for c in key):
        return None
    problem = _public_shape_problem(
        view, schema, key, listed_by, levels, candidates, aliases, numeric, pairs, values, metric_levels
    )
    if problem is not None:
        # Never a value in the log: only which rule refused it.
        logger.info("estimate_table_refused_not_public_shape", rule=problem, key_columns=len(key))
        return None
    dimensions = []
    for column in key:
        listed = listed_by[column]
        parents, top = parent_levels(listed, column)
        dimensions.append(
            Dimension(
                name=column,
                levels=levels[column],
                parents=parents,
                top=top,
                yearlike=_is_year(column, listed),
                aliases=tuple(aliases.get(column, ())),
                values=tuple(sorted(listed, key=str)),
            )
        )
    # An empty level alone (a total row) is not evidence of estimates: patient files have gaps too.
    has_totals = any(d.top is not None or any(p is not None for p in d.parents) for d in dimensions)
    # One row per state, county or local jurisdiction (and year...): rows keyed by place are
    # aggregates by construction; a patient file is never unique by place.
    by_place = any(_mostly_places(d.values, d.name) for d in dimensions)
    if not (has_totals or by_place or any(v.lower or v.kind == "rate" for v in values)):
        return None  # no interval, rate, total or state key: nothing says these rows are estimates
    attributes = tuple(c for c in kept if c not in key)
    nesting = {d.name: found for d in dimensions if (found := _county_states(view, d))}
    table = EstimateTable(
        rows=rows,
        dimensions=tuple(dimensions),
        values=values,
        attributes=attributes,
        metric_levels=metric_levels,
        nesting=nesting,
    )
    logger.info(
        "estimate_table_detected",
        dimensions=[d.name for d in table.dimensions],
        values=[v.name for v in table.values if v.lower],
        attributes=len(attributes),
    )
    return table


# ---------------------------------------------------------------------------- plan checks


def _level_text(level: Any) -> str:
    return "(empty: a total row)" if level is None else str(level)


def _plan_values_by_dimension(plan: QueryPlan, table: EstimateTable) -> dict[str, list[Any]]:
    """dimension -> the levels the plan's equality and IN limits keep (aliases folded in)."""
    found: dict[str, list[Any]] = {}
    limits: list[tuple[str, Any]] = [
        (spec.column, spec.value) for spec in plan.filters if spec.operator in ("==", "IN")
    ]
    limits += [(step.column, step.value) for step in plan.cohort_steps if step.op == "eq"]
    for column, value in limits:
        dimension = table.dimension(column)
        if dimension is None:
            continue
        items = value if isinstance(value, list) else [value]
        found.setdefault(dimension.name, []).extend(items)
    return found


def _open_text(dimension: Dimension, present: Sequence[Any]) -> str:
    """'sex (Both, Female, Male; Both is the total)' for a dimension the question leaves open.
    Only reached on a declared dataset, whose level labels are published categories."""
    if dimension.yearlike and present:
        numbers = sorted(present, key=str)
        listed = f"{len(present)} values, {numbers[0]}–{numbers[-1]}"
    elif len(present) <= config.ESTIMATE_LEVELS_IN_HOLD:
        listed = ", ".join(_level_text(level) for level in sorted(present, key=str))
    else:
        listed = f"{len(present)} values"
    role = "the summary" if _is_standardized_level(dimension.top) else "the total"
    total = f"; {dimension.top} is {role}" if dimension.top is not None and dimension.top in present else ""
    return f"{dimension.name} ({listed}{total})"


def check_plan(
    plan: QueryPlan,
    view: Any,
    table: EstimateTable,
    choices: Mapping[str, Any],
) -> EstimateCheck:
    """Refuse, hold or clear a statistic question on an estimate table.

    ``view`` is the base view with the plan's limits already applied (group column included);
    ``choices`` are the levels the person chose for this dataset version earlier.
    """
    if plan.intent in _PATIENT_TESTS:
        return EstimateCheck(refusal=config.ESTIMATE_PATIENT_TEST_REFUSAL)
    if plan.proportion_column or plan.rate_column:
        return EstimateCheck(refusal=config.ESTIMATE_SHARE_REFUSAL)
    value = table.value(plan.metric) if plan.metric else table.primary_value()
    if value is None:
        dimension = table.dimension(plan.metric)
        if dimension is not None:
            return EstimateCheck(refusal=config.ESTIMATE_KEY_NOT_VALUE.format(column=plan.metric, key=table.describe()))
        names = ", ".join(v.name for v in table.values[: config.ESTIMATE_LEVELS_IN_HOLD])
        return EstimateCheck(refusal=config.ESTIMATE_WHICH_VALUE.format(values=names))
    if plan.metric in (value.lower, value.upper):
        value = ValueColumn(str(plan.metric), None, None, value.kind)
    asked = _plan_values_by_dimension(plan, table)
    group_dims: list[str] = []
    grouped = table.dimension(plan.group_by)
    if grouped is not None:
        group_dims.append(grouped.name)
    elif plan.group_by:
        return EstimateCheck(refusal=config.ESTIMATE_GROUP_NOT_KEY.format(column=plan.group_by, key=table.describe()))
    group_dims += [dim for dim, items in asked.items() if len(set(items)) > 1 and dim not in group_dims]
    # Levels left after the limits, per dimension (one aggregate).
    dims = [d.name for d in table.dimensions]
    present_frame = view.aggregate(
        **{f"c{i}": view[d].collect(distinct=True, include_null=True) for i, d in enumerate(dims)}
    ).to_polars()
    collected = present_frame.row(0)
    present = {d: sorted(set(collected[i] or []), key=str) for i, d in enumerate(dims)}
    if not any(present.values()):
        return EstimateCheck(refusal=config.ESTIMATE_NO_ROWS)
    left_out: dict[str, tuple[Any, ...]] = {}
    pinned: dict[str, Any] = {}
    chosen: dict[str, Any] = {}
    open_dims: list[Dimension] = []
    for dimension in table.dimensions:
        levels = present[dimension.name]
        if dimension.name in group_dims:
            # A total beside its parts is never one of the groups compared or ranked.
            parts = [level for level in levels if level not in dimension.parents]
            if parts and len(parts) < len(levels):
                left_out[dimension.name] = tuple(level for level in levels if level in dimension.parents)
            continue
        if len(levels) == 1:
            pinned[dimension.name] = levels[0]
            continue
        saved = choices.get(dimension.name)
        if saved is not None and saved in levels:
            chosen[dimension.name] = saved
            pinned[dimension.name] = saved
            continue
        open_dims.append(dimension)
    check = EstimateCheck(value=value, group_dims=tuple(group_dims), pinned=pinned, left_out=left_out, chosen=chosen)
    if not open_dims:
        return check
    opened = "; ".join(_open_text(d, present[d.name]) for d in open_dims)
    without_total = [d.name for d in open_dims if d.top is None or d.top not in present[d.name]]
    if without_total:
        check.refusal = config.ESTIMATE_OPEN_REFUSAL.format(
            key=table.describe(), open=opened, name=", ".join(without_total)
        )
        return check
    check.defaults = {d.name: d.top for d in open_dims}
    totals = ", ".join(f"{name} = {level}" for name, level in check.defaults.items())
    check.hold = config.ESTIMATE_OPEN_HOLD.format(key=table.describe(), open=opened, totals=totals)
    return check


# ---------------------------------------------------------------------------- answers


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN is no value


def _fmt(value: float | None) -> str:
    if value is None:
        return "no value"
    return f"{value:,.2f}" if abs(value) < 1e6 else f"{value:,.0f}"


def _interval(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return ""
    return f" ({_fmt(lower)} to {_fmt(upper)})"


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool | None:
    """Whether two estimates' intervals overlap; None when either has no interval."""
    if a["lower"] is None or a["upper"] is None or b["lower"] is None or b["upper"] is None:
        return None
    return bool(a["lower"] <= b["upper"] and b["lower"] <= a["upper"])


def estimate_rows(
    view: Any,
    check: EstimateCheck,
    table: EstimateTable,
    *,
    lowest_first: bool,
) -> list[dict[str, Any]]:
    """One row per group: label, value, lower, upper (bounds None when the file has none).

    ``view`` has the plan's limits applied; the saved choices and the totals left out of a
    comparison are applied here. Groups are ordered by key for years, else by value.
    """
    value = check.value
    assert value is not None  # noqa: S101 - check_plan returned a value or a refusal
    for column, level in check.chosen.items():
        view = view.filter(view[column] == level)
    for column, levels in check.left_out.items():
        named = [level for level in levels if level is not None]
        if None in levels:
            view = view.filter(view[column].notnull())  # NOT IN with an empty value would drop every row
        if named:
            view = view.filter(~view[column].isin(named))
    metric_columns = [c for c in table.metric_levels if c in view.columns]
    key_columns = [d.name for d in table.dimensions]
    flags = [c for c in (value.suppressed, value.unreliable) if c]
    selected = list(
        dict.fromkeys(
            [
                *check.group_dims,
                *key_columns,
                *metric_columns,
                value.name,
                *(c for c in (value.lower, value.upper) if c),
                *flags,
            ]
        )
    )
    frame = view.select(*selected).to_polars()
    rows: list[dict[str, Any]] = []
    for record in frame.iter_rows(named=True):
        label = " · ".join(_level_text(record[d]) for d in check.group_dims)
        rows.append(
            {
                "label": label,
                "key": tuple(record[d] for d in check.group_dims),
                "value": _num(record[value.name]),
                "lower": _num(record[value.lower]) if value.lower else None,
                "upper": _num(record[value.upper]) if value.upper else None,
                "suppressed": bool(value.suppressed and record[value.suppressed]),
                "unreliable": bool(value.unreliable and record[value.unreliable]),
            }
        )
    for row in rows:
        if row["suppressed"]:
            # Suppressed at the source: never shown, never ordered by a value it doesn't have.
            row["value"] = row["lower"] = row["upper"] = None
    for row in rows:
        # lower <= value <= upper, or the interval is not the value's: dropped and named, never used
        # to call two estimates different (a row with one bound missing has no interval either).
        value_, lower, upper = row["value"], row["lower"], row["upper"]
        row["bad_interval"] = (
            value_ is not None and lower is not None and upper is not None and not lower <= value_ <= upper
        )
        if row["bad_interval"] or lower is None or upper is None:
            row["lower"] = row["upper"] = None
    if len({row["key"] for row in rows}) != len(rows):
        raise EstimateGrainError(config.ESTIMATE_DUPLICATE_ROWS)
    yearlike = bool(check.group_dims) and all(d.yearlike for d in table.dimensions if d.name in check.group_dims)
    for row in rows:
        row["ordered_by_key"] = yearlike  # the claim reads first and last, never "highest"
    if yearlike:
        rows.sort(key=lambda r: tuple(str(k) for k in r["key"]))
    else:
        rows.sort(
            key=lambda r: (
                r["value"] is None,
                0.0 if r["value"] is None else (r["value"] if lowest_first else -r["value"]),
                r["label"] if r["value"] is None else "",
            )
        )
    return rows


def _scope(check: EstimateCheck) -> str:
    parts = [f"{column} = {level}" for column, level in check.pinned.items()]
    return f" for {', '.join(parts)}" if parts else ""


def estimate_answer(
    rows: list[dict[str, Any]], check: EstimateCheck, *, compare: bool, lowest_first: bool
) -> dict[str, Any]:
    """The analysis dict for an estimate answer (``type: estimates``): claim and table ready to
    show. Values are shown as published; cells the source suppressed stay hidden; intervals are
    compared, never tested."""
    value = check.value
    assert value is not None  # noqa: S101 - check_plan returned a value or a refusal
    name = short_name(value.name)
    has_bounds = bool(value.lower and value.upper)
    table_rows: list[dict[str, Any]] = []
    group_header = " · ".join(check.group_dims) or "estimate"
    source_suppressed = 0
    for row in rows:
        shown: dict[str, Any] = {group_header: row["label"] or name}
        # Published estimates are shown as published (owner's decision, ADR014 §7); only what the
        # source itself suppressed stays hidden, and is never turned into a number.
        if row.get("suppressed"):
            source_suppressed += 1
            shown |= {name: SECONDARY_SUPPRESSED}
            if has_bounds:
                shown |= {"lower": SECONDARY_SUPPRESSED, "upper": SECONDARY_SUPPRESSED}
            row["value"] = row["lower"] = row["upper"] = None
        else:
            shown |= {name: row["value"]}
            if has_bounds:
                shown |= {"lower": row["lower"], "upper": row["upper"]}
        table_rows.append(shown)
    notes: list[str] = []
    for column, levels in check.left_out.items():
        listed = ", ".join(_level_text(level) for level in levels)
        notes.append(config.ESTIMATE_TOTAL_LEFT_OUT.format(levels=listed, column=column))
    for column, level in check.chosen.items():
        notes.append(config.ESTIMATE_SAVED_CHOICE_NOTE.format(column=column, level=level))
    if source_suppressed:
        notes.append(config.ESTIMATE_SOURCE_SUPPRESSED_NOTE)
    unreliable = [str(r["label"] or name) for r in rows if r.get("unreliable") and r["value"] is not None]
    if unreliable:
        notes.append(
            config.ESTIMATE_UNRELIABLE_NOTE.format(groups=", ".join(unreliable[: config.ESTIMATE_LEVELS_IN_HOLD]))
        )
    inconsistent = [str(r["label"] or name) for r in rows if r.get("bad_interval")]
    if inconsistent:
        notes.append(
            config.ESTIMATE_BAD_INTERVAL_NOTE.format(groups=", ".join(inconsistent[: config.ESTIMATE_LEVELS_IN_HOLD]))
        )
    if not has_bounds:
        notes.append(config.ESTIMATE_NO_BOUNDS_NOTE)
    shown_rows = [r for r in rows if r["value"] is not None]
    claim = _claim(name, shown_rows, check, compare=compare, lowest_first=lowest_first, has_bounds=has_bounds)
    if compare:
        notes.append(config.ESTIMATE_NO_TEST_NOTE)
    return {
        "type": "estimates",
        "claim": " ".join([claim, *notes]),
        "table": {"columns": list(table_rows[0]) if table_rows else [], "rows": table_rows} if table_rows else None,
        "value_column": value.name,
        "groups": len(rows),
        "unit": "estimates",
    }


def _claim(
    name: str,
    rows: list[dict[str, Any]],
    check: EstimateCheck,
    *,
    compare: bool,
    lowest_first: bool,
    has_bounds: bool,
) -> str:
    scope = _scope(check)
    if not rows:
        return f"No {name} value can be shown{scope}."
    if not check.group_dims:
        row = rows[0]
        return f"{name}{scope}: {_fmt(row['value'])}{_interval(row['lower'], row['upper'])}."
    by = " · ".join(check.group_dims)
    # The same year-like flag the rows were ordered by (a text season is year-like; an integer
    # code that is not a year is not), so the claim never calls the earliest row the highest.
    yearlike = bool(rows[0].get("ordered_by_key")) if "ordered_by_key" in rows[0] else _years(rows)
    if yearlike:
        first, last = rows[0], rows[-1]
        return (
            f"{name} by {by}{scope}: {_fmt(first['value'])}{_interval(first['lower'], first['upper'])} in "
            f"{first['label']} and {_fmt(last['value'])}{_interval(last['lower'], last['upper'])} in "
            f"{last['label']} ({len(rows)} values; all in the table)."
        )
    if len(rows) == 2:
        a, b = rows
        verdict = _two_way(a, b, has_bounds)
        return (
            f"{name} by {by}{scope}: {a['label']} {_fmt(a['value'])}{_interval(a['lower'], a['upper'])}, "
            f"{b['label']} {_fmt(b['value'])}{_interval(b['lower'], b['upper'])}. {verdict}"
        )
    top = rows[0]
    word = "Lowest" if lowest_first else "Highest"
    claim = (
        f"{name} by {by}{scope}, {len(rows)} groups. {word}: {top['label']} "
        f"{_fmt(top['value'])}{_interval(top['lower'], top['upper'])}."
    )
    if not has_bounds:
        return claim
    overlapping = [r["label"] for r in rows[1:] if _overlap(top, r)]
    if overlapping:
        listed = ", ".join(overlapping[: config.ESTIMATE_LEVELS_IN_HOLD])
        more = (
            f" and {len(overlapping) - config.ESTIMATE_LEVELS_IN_HOLD} more"
            if len(overlapping) > config.ESTIMATE_LEVELS_IN_HOLD
            else ""
        )
        claim += f" Its interval overlaps those of {listed}{more}, so its rank among them is uncertain."
    else:
        claim += " Its interval does not overlap any other group's."
    return claim


def _years(rows: list[dict[str, Any]]) -> bool:
    return all(len(r["key"]) == 1 and isinstance(r["key"][0], int) for r in rows)


def _two_way(a: Mapping[str, Any], b: Mapping[str, Any], has_bounds: bool) -> str:
    if not has_bounds:
        return "This file gives no intervals, so whether they differ can't be judged."
    overlap = _overlap(a, b)
    if overlap is None:
        return "One of them has no interval, so whether they differ can't be judged."
    if overlap:
        return "Their intervals overlap, so these estimates are not shown to differ."
    return "Their intervals don't overlap, so the estimates differ."


_TREND_WORDS = re.compile(
    r"\b(by year|each year|per year|every year|yearly|annual|annually|over time|trend|trends|time series|"
    r"over the years|by season|each season)\b",
    re.IGNORECASE,
)
_RANGE_WORDS = re.compile(r"\b(from|between|since|change|changed|compare|compared|versus|vs)\b", re.IGNORECASE)
_RANK_WORDS = re.compile(r"\b(which|rank|ranking|highest|lowest|top|bottom|most|least)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\b(\d{4})\b")
_SUBAREA_WORDS: Final = frozenset({"county", "countie", "parish", "borough"})


def _words(text: str) -> list[str]:
    """Lowercase words with a plural "s" dropped ("diseases" -> "disease", "males" -> "male")."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return [_singular(w) for w in words]


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _phrase_in(level: str, question: list[str]) -> bool:
    phrase = _words(level)
    if not phrase:
        return False
    width = len(phrase)
    return any(question[i : i + width] == phrase for i in range(len(question) - width + 1))


def _named_levels(dimension: Dimension, question: list[str]) -> list[Any]:
    """The dimension's text levels the question names as a phrase; a level inside a longer
    named level ("Male" inside nothing, "disease" inside "heart disease") gives way to it."""
    named = [level for level in dimension.values if isinstance(level, str) and _phrase_in(level, question)]
    return [level for level in named if not any(other != level and _phrase_in(level, _words(other)) for other in named)]


def ground_plan(plan: QueryPlan, table: EstimateTable, question: str) -> QueryPlan:
    """``plan`` with the key dimensions the question names but the parse left open (ADR014).

    Deterministic, from the question's words and the dimensions' published levels only:

    * a year (or several) the question names, for a year-like dimension: one pins it; two or more
      with "from/between/change/compare" keep those years as groups;
    * "by year", "trend", "over time": group by the year-like dimension;
    * a level named as a phrase ("cardiovascular disease" names "Cardiovascular diseases"; "males
      and females" names Male and Female): one pins it, several keep them as groups;
    * a total level named next to "county"/"counties" and a ranking word ("Which Colorado county
      had the highest ...") groups by that dimension instead, the total left out.

    Dimensions the plan already limits or groups by are left as parsed.
    """
    if plan.intent not in _STATISTIC_INTENTS:
        return plan
    words = _words(question)
    limited = set(_plan_values_by_dimension(plan, table))
    limited |= {d.name for spec in plan.filters if (d := table.dimension(spec.column)) is not None}
    grouped = table.dimension(plan.group_by)
    filters = list(plan.filters)
    steps = list(plan.cohort_steps)
    group_by = plan.group_by
    for dimension in table.dimensions:
        if dimension.levels < 2 or (grouped is not None and dimension.name == grouped.name):
            continue
        named = _named_levels(dimension, words) if not dimension.yearlike else []
        if group_by is None and _ranks_subareas(dimension, named, question, words, table):
            # "Which Colorado county had the highest ...": rank Colorado's own counties, not the
            # state row, and never another state's counties.
            group_by = dimension.name
            filters = [f for f in filters if table.dimension(f.column) is not dimension]
            steps = [st for st in steps if table.dimension(st.column) is not dimension]
            nested = table.nesting.get(dimension.name, {})
            own = [county for county, state in nested.items() if state in named]
            if own:
                filters.append(FilterSpec(column=dimension.name, operator="IN", value=sorted(own, key=str)))
            continue
        if dimension.name in limited:
            if named and not _limits_within(plan, table, dimension, named):
                # The question names levels the parse did not keep ("India and Indonesia" read
                # as another country): the question's own words win.
                filters = [f for f in filters if table.dimension(f.column) is not dimension]
                steps = [st for st in steps if table.dimension(st.column) is not dimension]
                filters.append(_limit(dimension, named))
            continue
        if group_by is None and not named and _ranks_by_name(dimension, question, words):
            group_by = dimension.name  # "Which countries had the highest ..."
            continue
        if dimension.yearlike:
            years = [int(y) for y in _NUMBER.findall(question)]
            present = [y for y in dict.fromkeys(years) if y in dimension.values or str(y) in dimension.values]
            typed: list[str | int | float] = [y if y in dimension.values else str(y) for y in present]
            if len(typed) == 1:
                filters.append(FilterSpec(column=dimension.name, operator="==", value=typed[0]))
            elif len(typed) > 1 and _RANGE_WORDS.search(question):
                filters.append(FilterSpec(column=dimension.name, operator="IN", value=typed))
            elif not typed and group_by is None and _TREND_WORDS.search(question):
                group_by = dimension.name
            continue
        if named:
            filters.append(_limit(dimension, named))
    metric = plan.metric if table.value(plan.metric) is not None else None
    named_value = named_value_column(table, question)
    if named_value is not None:
        metric = named_value.name  # the question's exact column name wins over a fuzzy parse
    elif metric is None:
        primary = table.primary_value()
        metric = primary.name if primary else plan.metric
    if filters == plan.filters and steps == plan.cohort_steps and group_by == plan.group_by and metric == plan.metric:
        return plan
    return dataclasses.replace(plan, filters=filters, cohort_steps=steps, group_by=group_by, metric=metric)


def _limit(dimension: Dimension, levels: list[Any]) -> FilterSpec:
    if len(levels) == 1:
        return FilterSpec(column=dimension.name, operator="==", value=levels[0])
    return FilterSpec(column=dimension.name, operator="IN", value=list(levels))


def _limits_within(plan: QueryPlan, table: EstimateTable, dimension: Dimension, named: list[Any]) -> bool:
    """Every level the plan keeps on ``dimension`` is one the question names."""
    kept = _plan_values_by_dimension(plan, table).get(dimension.name, [])
    return all(level in named for level in kept)


def named_value_column(table: EstimateTable, question: str) -> ValueColumn | None:
    """The value column the question names by its exact name ("e_inc_100k"), longest first."""
    text = question.casefold()
    named = [
        value for value in table.values if re.search(rf"(?<![\w]){re.escape(value.name.casefold())}(?![\w])", text)
    ]
    return max(named, key=lambda v: len(v.name)) if named else None


def _ranks_by_name(dimension: Dimension, question: str, words: list[str]) -> bool:
    """The question ranks the dimension's levels and names the dimension ("which countries")."""
    name = _words(dimension.name.replace("_name", ""))
    return bool(name) and _phrase_in(" ".join(name), words) and bool(_RANK_WORDS.search(question))


def _ranks_subareas(
    dimension: Dimension, named: list[Any], question: str, words: list[str], table: EstimateTable
) -> bool:
    """The question ranks the county-like levels under a total it names."""
    return (
        bool(named)
        and all(level in dimension.parents for level in named)
        and any(w in _SUBAREA_WORDS for w in words)
        and bool(_RANK_WORDS.search(question))
        and any(isinstance(v, str) and config.ESTIMATE_SUBAREA_NAME.search(v) for v in dimension.values)
        # Counties of one state only, or every county known to nest under its state: otherwise a
        # national county file would rank every state's counties under "Colorado", or leave out
        # the counties whose state is not known (a partial ranking read as the whole state's).
        and _nesting_complete(dimension, table)
    )


def _subareas(dimension: Dimension) -> list[Any]:
    return [v for v in dimension.values if isinstance(v, str) and config.ESTIMATE_SUBAREA_NAME.search(v.split(",")[0])]


def _nesting_complete(dimension: Dimension, table: EstimateTable) -> bool:
    """Every county-like level's state is known, or the dimension holds one state at most."""
    if sum(_place_kind(v, dimension.name) == "state" for v in dimension.parents) <= 1:
        return True
    nested = table.nesting.get(dimension.name, {})
    return all(level in nested for level in _subareas(dimension))


def subarea_ranking_refusal(table: EstimateTable, question: str, name: str) -> str | None:
    """The refusal for "Which Texas county ..." on a file of several states' counties whose
    counties are not all known to nest under their state, or None. Ranking only the counties
    whose state is known would read as the whole state's ranking; ranking all would mix states."""
    words = _words(question)
    if not _RANK_WORDS.search(question) or not any(w in _SUBAREA_WORDS for w in words):
        return None
    for dimension in table.dimensions:
        if dimension.yearlike or not _subareas(dimension):
            continue
        named = _named_levels(dimension, words)
        if named and all(level in dimension.parents for level in named) and not _nesting_complete(dimension, table):
            return config.ESTIMATE_SUBAREA_NESTING_UNKNOWN.format(source=name, column=dimension.name)
    return None


def grounded_words(table: EstimateTable, question: str) -> str:
    """``question`` without the words estimate questions use for their own structure (years,
    named levels, "rate", "highest", "county"): what is left must still be read by the plan
    (word accounting runs on it). A word the plan leaves unused outside these is still held."""
    named_value = named_value_column(table, question)
    if named_value is not None:
        question = re.sub(re.escape(named_value.name), " ", question, flags=re.IGNORECASE)
    words = question.split()
    level_words: set[str] = set()
    for dimension in table.dimensions:
        for level in dimension.values:
            if isinstance(level, str) and _phrase_in(level, _words(question)):
                level_words |= set(_words(level))
    kept = []
    for word in words:
        normalised = _words(word)
        if normalised and all(
            w in level_words or w in config.ESTIMATE_QUESTION_WORDS or _NUMBER.fullmatch(w) for w in normalised
        ):
            continue
        kept.append(word)
    return " ".join(kept)


def lowest_first(question: str) -> bool:
    """The question asks for the lowest values first ("lowest", "least", ...)."""
    return bool(_LOWEST_WORDS.search(question))


def unlimited_group_plan(plan: QueryPlan) -> QueryPlan:
    """``plan`` with no grouping: the filtered view then keeps limits on the group column too."""
    return dataclasses.replace(plan, group_by=None)


def out_wayne_refusal(table: EstimateTable, question: str, name: str) -> str | None:
    """The refusal for a question about Wayne County on a table whose Wayne rows are out-Wayne
    (Wayne County excluding Detroit, which reports separately), or None."""
    from clinical_analytics.contracts import geography

    if not geography.mentions_wayne(question) or re.search(r"out[- ]?wayne", question, re.IGNORECASE):
        return None
    levels = [v for d in table.dimensions for v in d.values]
    if not any(geography.is_out_wayne(v) for v in levels):
        return None
    has_detroit = any(geography.is_detroit(v) for v in levels)
    return config.ESTIMATE_JOIN_OUT_WAYNE.format(
        source=name, detroit="It also has Detroit rows" if has_detroit else "It has no Detroit rows"
    )


def detect_frame(frame: pl.DataFrame, skip: Sequence[str] = ()) -> EstimateTable | None:
    """``detect_estimate_table`` on an in-memory frame (one stored table of an upload), through a
    private in-memory DuckDB so detection is the same code the question path runs."""
    import ibis

    con = ibis.duckdb.connect()
    try:
        con.con.register("estimate_frame", frame.to_arrow())
        return detect_estimate_table(con.table("estimate_frame"), frame.height, skip)
    finally:
        con.disconnect()
