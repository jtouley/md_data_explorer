# SPDX-License-Identifier: MIT
"""Questions that join several public aggregate tables (ADR014 §10).

A fellow uploads the public tables a paper needs (a GBD extract, a CDC WONDER export, AtlasPlus,
PLACES, SVI, FluVaxView), each declared public population estimates, and asks one plain-English
question across them. This module answers it, holds it, or refuses it:

1. **Per table, pin everything the join does not key on.** Each table is an estimate table
   (``estimates.detect_estimate_table``). Every key dimension that is not a join role (cause,
   measure, metric, ICD code) must come down to one level: named in the question, chosen by the
   person, or offered as a hold when the dimension has a total. Otherwise the question is refused
   and the refusal lists the levels.
2. **Harmonize the join keys.** State names, USPS codes and FIPS numbers become a two-digit FIPS
   text key (``contracts/geography.py``); sex labels become Female, Male or Both; years become
   integers. Levels that are not states (the nation, counties, Detroit, out-Wayne) are left out and
   named. Rows with an empty key (WONDER's "Total" rows) are left out and counted.
3. **Check the grain.** Each table must have exactly one row per harmonized key after pinning
   (GX ``expect_compound_columns_to_be_unique``); otherwise nothing is answered (fail closed).
4. **Join** on the keys the tables share, inner, and say how many keys were only in one table.
5. **Answer** with one of five analyses, each with its method written out:

   * ``table``: the values side by side, nothing computed across tables;
   * ``change``: each year's value with that year's own interval, the change between the first and
     last year, and never an interval for the change built from the years' intervals;
   * ``rank``: rank among states, with the range of ranks the intervals allow;
   * ``disagree``: where a reported value falls outside a modelled estimate's interval;
   * ``precision``: interval width relative to the estimate, widest first.

Comparisons across tables need the same kind of number (rate, count, percent), the same scale and
the same standardization: an age-standardized rate is never compared with a crude one, and two
age-standardized rates only when they use the same standard population.

Privacy: every table must be declared public population estimates for its current version. Counts
(a count column, or a row whose metric is "Number") keep the small-cell rule, so a count from 1 to
10 is never shown and a change or rank built on one is not shown either. Cells the source
suppressed stay suppressed.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

import polars as pl
import structlog

from clinical_analytics.contracts import config, geography
from clinical_analytics.core import estimates
from clinical_analytics.core.estimates import Dimension, EstimateTable, ValueColumn
from clinical_analytics.core.result_presentation import SECONDARY_SUPPRESSED

logger = structlog.get_logger()

Role: TypeAlias = Literal["state", "year", "season", "sex", "age"]
Analysis: TypeAlias = Literal["table", "change", "rank", "disagree", "precision"]
Status: TypeAlias = Literal["answered", "held", "refused"]
Standardization: TypeAlias = Literal["age-standardized", "crude", "unknown", "not a rate"]

# Request choices that are not a column level: which value column, and a rate's standardization.
VALUE_CHOICE: Final = "__value__"
STANDARDIZATION_CHOICE: Final = "__standardization__"
STANDARDIZATION_LEVELS: Final = ("age-standardized", "crude")

_SEASON: Final = re.compile(r"^\s*(\d{4})\s*[-–/]\s*(\d{2}|\d{4})\s*$")
_YEAR: Final = re.compile(r"\b(19\d\d|20\d\d)\b")
_SPAN: Final = re.compile(r"\b(19\d\d|20\d\d)\s*(?:-|–|—|to|through|until)\s*(19\d\d|20\d\d)\b", re.IGNORECASE)
_PER: Final = re.compile(r"per[\s_]*([\d,]+)\s*(k?)", re.IGNORECASE)
_KEYS: Final = ("state", "year", "sex", "age")
_STATE_NAME: Final = "state"
_STATE_FIPS: Final = "state_fips"
# Per table on each harmonized row: value, lower, upper, suppressed at source, unreliable, a count.
# Reasons a cell is never shown (a value with a bad interval is shown, without the interval).
_HIDDEN: Final = ("source", "small", "withheld")
_VALUE_COLUMNS: Final = ("_v", "_lo", "_hi", "_sup", "_unr", "_cnt", "_wh")


class JoinRefusedError(Exception):
    """A question that can't be answered as asked; the message says why and what to change."""


@dataclass(frozen=True)
class SourceInput:
    """One uploaded table as the API loads it (its active version)."""

    dataset_id: str
    name: str
    frame: pl.DataFrame
    table: EstimateTable | None
    declared: bool
    saved_choices: Mapping[str, Any] = field(default_factory=dict)
    version: str | None = None  # the dataset version read: choices confirmed on it are saved to it


@dataclass(frozen=True)
class HoldOption:
    """One choice the person confirms to release a held question."""

    dataset_id: str
    dataset_name: str
    column: str
    default: Any
    levels: tuple[Any, ...]


@dataclass
class JoinAnswer:
    status: Status
    message: str
    table: dict[str, Any] | None = None
    method: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    options: list[HoldOption] = field(default_factory=list)


@dataclass(frozen=True)
class Measure:
    kind: estimates.ValueKind | Literal["percent"]
    standardization: Standardization
    standard: str | None  # standard population, when known ("GBD world population")
    per: int | None  # rate scale (100000), when known


@dataclass
class Asked:
    """What the question's own words fix (deterministic; no model)."""

    text: str
    words: list[str]
    analysis: Analysis
    states: list[str]  # FIPS codes named (states and named groups)
    focus: list[str]  # FIPS codes of the states named one by one
    groups: list[str]  # named groups of states ("great lakes")
    all_states: bool
    nation: bool
    years: list[int]
    trend: bool
    sexes: list[str]
    by_sex: bool
    standardization: Literal["age-standardized", "crude"] | None
    lowest_first: bool


@dataclass
class Prepared:
    """One table after pinning: its harmonized rows and what the answer says about it."""

    source: SourceInput
    value: ValueColumn
    measure: Measure
    frame: pl.DataFrame  # harmonized: keys + _VALUE_COLUMNS
    keys: list[str]
    groups: list[str]  # non-role dimensions grouped (one-table questions only)
    pinned: dict[str, Any]
    method: str
    notes: list[str]


# ---------------------------------------------------------------------------- question words


def _words(text: str) -> list[str]:
    expanded = text.casefold()
    for short, long in config.ESTIMATE_QUESTION_SYNONYMS.items():
        expanded = re.sub(rf"(?<![\w]){re.escape(short)}(?![\w])", f"{short} {long}", expanded)
    return estimates._words(expanded)


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def read_question(question: str) -> Asked:
    """The analysis, states, years, sexes and standardization the question names."""
    text = " ".join(question.split())
    words = _words(text)
    if _has(text, config.ESTIMATE_JOIN_PRECISION_WORDS):
        analysis: Analysis = "precision"
    elif _has(text, config.ESTIMATE_JOIN_DISAGREE_WORDS):
        analysis = "disagree"
    elif _has(text, config.ESTIMATE_JOIN_RANK_WORDS):
        analysis = "rank"
    elif _has(text, config.ESTIMATE_JOIN_CHANGE_WORDS):
        analysis = "change"
    else:
        analysis = "table"
    focus = [s.fips for s in geography.state_named(text)]
    groups = geography.groups_named(text)
    states = list(focus)
    for group in groups:
        states += [f for f in geography.STATE_GROUPS[group] if f not in states]
    years: list[int] = []
    trend = _has(text, config.ESTIMATE_JOIN_TREND_WORDS)
    for start, end in _SPAN.findall(text):
        low, high = sorted((int(start), int(end)))
        years += [low, high]
        trend = trend or not _has(text, r"\b(change|changed)\b")
    years += [int(y) for y in _YEAR.findall(text)]
    years = sorted(dict.fromkeys(years))
    sexes = [
        canonical
        for canonical, pattern in config.ESTIMATE_JOIN_SEX_WORDS.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]
    by_sex = _has(text, r"\bby (sex|gender)\b") or len(sexes) > 1
    found = {std for _, std in _standardization_phrases(text)}
    standardization: Literal["age-standardized", "crude"] | None = found.pop() if len(found) == 1 else None
    return Asked(
        text=text,
        words=words,
        analysis=analysis,
        states=states,
        focus=focus,
        groups=groups,
        all_states=_has(text, config.ESTIMATE_JOIN_ALL_STATES_WORDS),
        nation=_has(text, r"\b(united states|nationally|national|the nation)\b")
        or bool(re.search(r"\b(US|U\.S\.|USA)\b", text)),
        years=years,
        trend=trend,
        sexes=sexes,
        by_sex=by_sex,
        standardization=standardization,
        lowest_first=estimates.lowest_first(text),
    )


_STANDARDIZATION_PHRASE: Final = re.compile(
    r"\b(?P<std>age[- ]?(?:standardi[sz]ed|adjusted))\b|\b(?P<crude>crude|all ages)\b", re.IGNORECASE
)


def _standardization_phrases(text: str) -> list[tuple[int, Literal["age-standardized", "crude"]]]:
    return [
        (m.start(), "age-standardized" if m.group("std") else "crude") for m in _STANDARDIZATION_PHRASE.finditer(text)
    ]


def standardization_by_source(
    sources: Sequence[SourceInput], text: str
) -> dict[str, Literal["age-standardized", "crude"] | None]:
    """Which standardization the question asks of each table. "WONDER crude ... GBD
    age-standardized" gives WONDER crude and GBD age-standardized: each phrase belongs to the table
    named just before it (or, before any table is named, to the first one named after it). One kind
    of phrase, or tables the question doesn't name, share it. A named table with no phrase of its
    own gets none, so its age choice is held or refused, never borrowed from another table."""
    phrases = _standardization_phrases(text)
    kinds = {std for _, std in phrases}
    mentions = sorted(
        (m.start(), s.dataset_id)
        for s in sources
        for m in re.finditer(rf"(?<![\w]){re.escape(s.name)}(?![\w])", text, re.IGNORECASE)
    )
    if len(kinds) <= 1 or len(sources) < 2 or not mentions:
        shared = next(iter(kinds)) if len(kinds) == 1 else None
        return {s.dataset_id: shared for s in sources}
    chosen: dict[str, Literal["age-standardized", "crude"] | None] = {s.dataset_id: None for s in sources}
    for position, std in phrases:
        before = [dataset_id for start, dataset_id in mentions if start <= position]
        owner = before[-1] if before else mentions[0][1]
        chosen[owner] = chosen[owner] or std
    return chosen


# ---------------------------------------------------------------------------- roles


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def _season_levels(levels: Sequence[Any]) -> bool:
    texts = [v for v in levels if v is not None]
    return bool(texts) and all(isinstance(v, str) and _SEASON.match(v) for v in texts)


def role_of(dimension: Dimension) -> Role | None:
    """The join role a key dimension plays, or None for a dimension the join never keys on."""
    norm = _norm(dimension.name)
    parts = norm.split("_")
    levels = [v for v in dimension.values if v is not None]
    if _season_levels(levels) or "season" in parts:
        return "season"
    if dimension.yearlike:
        return "year"
    if {"sex", "gender"} & set(parts) or (
        levels and all(isinstance(v, str) and v.strip().casefold() in config.ESTIMATE_JOIN_SEX_LEVELS for v in levels)
    ):
        return "sex"
    if "age" in parts or norm.startswith("age"):
        return "age"
    if levels:
        codes = geography.is_geo_code_column(dimension.name)
        kinds = [geography.classify(v, codes).kind for v in levels]
        if kinds.count("state") >= max(1, len(levels) // 2):
            return "state"
    return None


def _sex(level: Any) -> str | None:
    if not isinstance(level, str):
        return None
    return config.ESTIMATE_JOIN_SEX_LEVELS.get(level.strip().casefold())


# ---------------------------------------------------------------------------- measures


def measure_of(source: SourceInput, value: ValueColumn, pinned: Mapping[str, Any]) -> Measure:
    """What kind of number the pinned value is: rate, count, percent; standardized or crude."""
    table = source.table
    assert table is not None  # noqa: S101 - callers refused undetected tables
    pinned_text = {_norm(str(v)) for v in pinned.values()}
    metric_levels = {_norm(str(level)) for column, level in pinned.items() if estimates.is_metric_column(column)}
    # Fail closed, as on the chat path: a metric level that says count makes it a count; otherwise
    # it is a rate (or a percent) only when its column name or a metric level positively says so.
    level_kinds = {
        estimates.metric_level_kind(level) for column, level in pinned.items() if estimates.is_metric_column(column)
    }
    kind: estimates.ValueKind | Literal["percent"]
    if "count" in level_kinds:
        kind = "count"
    elif "rate" in level_kinds or value.kind == "rate":
        percent = any(("percent" in t or "proportion" in t) for t in metric_levels) or _has(
            value.name, r"(pct|percent|%|proportion)"
        )
        kind = "percent" if percent else "rate"
    else:
        kind = "count"
    name = _norm(value.name)
    chosen = source.saved_choices.get(STANDARDIZATION_CHOICE)
    standardization: Standardization = "unknown"
    standard: str | None = None
    if kind not in ("rate", "value"):
        standardization = "not a rate"
    elif any(level in config.ESTIMATE_STANDARDIZED_LEVELS for level in pinned_text) or any(
        "age_adjusted" in level or "age_standardi" in level for level in pinned_text
    ):
        standardization = "age-standardized"
    elif "age_adjusted" in name or "age_standardi" in name:
        standardization = "age-standardized"
    elif "crude" in name or any("crude" in level or level == "all_ages" for level in pinned_text):
        standardization = "crude"
    elif chosen in STANDARDIZATION_LEVELS:
        standardization = chosen
    if standardization == "age-standardized" and any(
        _norm(d.name) == "age_name" or "Age-standardized" in d.values for d in table.dimensions
    ):
        standard = "GBD world population"
    per: int | None = None
    match = _PER.search(value.name)
    if match:
        per = int(match.group(1).replace(",", "")) * (1000 if match.group(2) else 1)
    elif kind == "rate" and (
        metric_levels & {"rate"} or value.name.strip().casefold() in config.ESTIMATE_JOIN_PER_100K_COLUMNS
    ):
        per = 100_000  # GBD "Rate" and CDC WONDER's rate columns are per 100,000
    return Measure(kind=kind, standardization=standardization, standard=standard, per=per)


def _standardization_of_name(name: str) -> str | None:
    norm = _norm(name)
    if "age_adjusted" in norm or "age_standardi" in norm:
        return "age-standardized"
    if "crude" in norm:
        return "crude"
    return None


def _describe_measure(measure: Measure) -> str:
    parts: list[str] = [str(measure.kind)]
    if measure.per:
        parts.append(f"per {measure.per:,}")
    if measure.standardization in ("age-standardized", "crude"):
        parts.append(measure.standardization)
    if measure.standard:
        parts.append(f"standard: {measure.standard}")
    return ", ".join(parts)


def comparable(a: Prepared, b: Prepared) -> str | None:
    """Why two tables' values can't be compared, or None when they can."""
    ma, mb = a.measure, b.measure
    names = (a.source.name, b.source.name)
    if ma.kind != mb.kind:
        return config.ESTIMATE_JOIN_KIND_MISMATCH.format(a=names[0], ak=ma.kind, b=names[1], bk=mb.kind)
    if ma.kind == "rate" and ma.per and mb.per and ma.per != mb.per:
        return config.ESTIMATE_JOIN_SCALE_MISMATCH.format(a=names[0], ap=ma.per, b=names[1], bp=mb.per)
    if ma.kind == "rate":
        if ma.standardization != mb.standardization:
            return config.ESTIMATE_JOIN_STANDARDIZATION_MISMATCH.format(
                a=names[0], ast=ma.standardization, b=names[1], bst=mb.standardization
            )
        if ma.standardization == "age-standardized" and (ma.standard is None or ma.standard != mb.standard):
            return config.ESTIMATE_JOIN_STANDARD_MISMATCH.format(a=names[0], b=names[1])
    return None


# ---------------------------------------------------------------------------- pinning


@dataclass
class _Pins:
    pinned: dict[str, Any] = field(default_factory=dict)
    kept: dict[str, list[Any]] = field(default_factory=dict)  # dimension -> levels kept (several)
    groups: list[str] = field(default_factory=list)
    holds: list[tuple[str, HoldOption]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    roles: dict[Role, str] = field(default_factory=dict)
    constants: dict[Role, tuple[str, Any]] = field(default_factory=dict)  # role -> (column, its one level)
    year_aligned: int | None = None  # a chosen year shown beside the question's year


def _choice(source: SourceInput, choices: Mapping[str, Any], column: str) -> Any:
    if column in choices:
        return choices[column]
    return source.saved_choices.get(column)


def _levels_text(dimension: Dimension, levels: Sequence[Any] | None = None) -> str:
    shown = list(levels if levels is not None else dimension.values)
    listed = ", ".join(str(v) for v in shown[: config.ESTIMATE_LEVELS_IN_HOLD])
    more = len(shown) - config.ESTIMATE_LEVELS_IN_HOLD
    return listed + (f" and {more} more" if more > 0 else "")


def _metric_named(dimension: Dimension, asked: Asked) -> list[Any]:
    """A metric level the question's words name ("rate" -> Rate, "how many" -> Number)."""
    if not config.ESTIMATE_METRIC_DIMENSION.search(_norm(dimension.name)):
        return []
    found = []
    for level in dimension.values:
        pattern = config.ESTIMATE_JOIN_METRIC_WORDS.get(_norm(str(level)))
        if pattern and re.search(pattern, asked.text, re.IGNORECASE):
            found.append(level)
    return found


def _wants_grouped(dimension: Dimension, asked: Asked) -> bool:
    """'by cause', 'each cause', 'all causes', 'which diseases' for a non-role dimension."""
    stem = _norm(dimension.name.replace("_name", "").replace("_id", "")).replace("_", " ")
    names = [stem, *config.ESTIMATE_JOIN_DIMENSION_WORDS.get(stem, ())]
    return any(
        re.search(rf"\b(by|each|every|all|which|across|and)\s+{re.escape(n)}(e?s)?\b", asked.text, re.IGNORECASE)
        for n in names
    )


def _pin_dimensions(source: SourceInput, asked: Asked, choices: Mapping[str, Any], single: bool) -> _Pins:
    table = source.table
    assert table is not None  # noqa: S101
    pins = _Pins()
    for dimension in table.dimensions:
        role = role_of(dimension)
        if role == "age":
            # Age is a join key only when the question asks "by age"; otherwise it is pinned like
            # any other dimension (to one band, "All ages" or "Age-standardized").
            if "age" not in pins.roles and _age_is_role(asked):
                pins.roles["age"] = dimension.name
            else:
                _pin_one(source, dimension, asked, choices, single, pins)
            continue
        if role is not None and role not in pins.roles:
            pins.roles[role] = dimension.name
            continue
        _pin_one(source, dimension, asked, choices, single, pins)
    for column, level in _constant_attributes(source).items():
        probe = Dimension(name=column, levels=1, values=(level,), yearlike=estimates._is_year(column, [level]))
        role = role_of(probe)
        if role is not None and role not in pins.roles and role not in pins.constants:
            pins.constants[role] = (column, level)
    return pins


def _age_is_role(asked: Asked) -> bool:
    return bool(re.search(r"\bby age\b", asked.text, re.IGNORECASE))


def _pin_one(
    source: SourceInput, dimension: Dimension, asked: Asked, choices: Mapping[str, Any], single: bool, pins: _Pins
) -> None:
    """Pin one dimension the join does not key on: a choice, a named level, a hold or a refusal."""
    chosen = _choice(source, choices, dimension.name)
    if chosen is not None and chosen in dimension.values:
        pins.pinned[dimension.name] = chosen
        return
    if dimension.levels == 1 and dimension.values:
        pins.pinned[dimension.name] = dimension.values[0]
        return
    named = estimates._named_levels(dimension, asked.words) or _metric_named(dimension, asked)
    if role_of(dimension) == "age" and not named and asked.standardization:
        named = [
            v
            for v in dimension.values
            if (asked.standardization == "age-standardized" and estimates._is_standardized_level(v))
            or (asked.standardization == "crude" and str(v).strip().casefold() == "all ages")
        ]
    if len(named) == 1:
        pins.pinned[dimension.name] = named[0]
        return
    grouped = single and asked.analysis in ("table", "rank", "precision") and _wants_grouped(dimension, asked)
    if grouped:
        pins.groups.append(dimension.name)
        if named:
            pins.kept[dimension.name] = list(named)
        return
    if len(named) > 1:
        pins.refusals.append(
            config.ESTIMATE_JOIN_SEVERAL_LEVELS.format(
                source=source.name, column=dimension.name, levels=_levels_text(dimension, named)
            )
        )
        return
    if dimension.top is not None:
        option = HoldOption(source.dataset_id, source.name, dimension.name, dimension.top, dimension.values)
        pins.holds.append(
            (
                config.ESTIMATE_JOIN_OPEN_HOLD.format(
                    source=source.name, column=dimension.name, levels=_levels_text(dimension), top=dimension.top
                ),
                option,
            )
        )
        return
    if role_of(dimension) == "age" and any(estimates._is_standardized_level(v) for v in dimension.values):
        pins.refusals.append(config.ESTIMATE_JOIN_AGE_OPEN.format(source=source.name))
        return
    pins.refusals.append(
        config.ESTIMATE_JOIN_OPEN_REFUSAL.format(
            source=source.name, column=dimension.name, levels=_levels_text(dimension)
        )
    )


def _choose_value(
    source: SourceInput, asked: Asked, choices: Mapping[str, Any]
) -> tuple[ValueColumn | None, HoldOption | None]:
    table = source.table
    assert table is not None  # noqa: S101
    chosen = _choice(source, choices, VALUE_CHOICE)
    if chosen is not None:
        found = table.value(chosen)
        if found is not None and found.name == chosen:
            return found, None
    named = estimates.named_value_column(table, asked.text)
    if named is not None:
        return named, None
    wants_rate = _has(asked.text, r"\b(rates?|per 100,?000|prevalence|percent|%)")
    phrased = [
        v for v in table.values if estimates._phrase_in(v.name, asked.words) and not (wants_rate and v.kind == "count")
    ]
    if phrased:
        longest = max(len(estimates._words(v.name)) for v in phrased)
        top = [v for v in phrased if len(estimates._words(v.name)) == longest]
        if len(top) == 1:
            return top[0], None
    primary = table.primary_value()
    if primary is not None:
        return primary, None
    wants_count = not wants_rate and _has(asked.text, r"\b(how many|number of|count|counts|deaths|cases)\b")
    by_kind = [v for v in table.values if (wants_count and v.kind == "count") or (wants_rate and v.kind == "rate")]
    if len(by_kind) > 1 and asked.standardization is not None:
        by_kind = [v for v in by_kind if _standardization_of_name(v.name) == asked.standardization]
    if len(by_kind) == 1:
        return by_kind[0], None
    names = tuple(v.name for v in table.values)
    # The default offered is of the kind the question asks for (a rate when it says "rate").
    fitting = by_kind or [v for v in table.values if v.lower and v.upper] or list(table.values)
    default = fitting[0].name
    return None, HoldOption(source.dataset_id, source.name, VALUE_CHOICE, default, names)


# ---------------------------------------------------------------------------- harmonize


def _state_map(levels: Sequence[Any]) -> tuple[pl.DataFrame, dict[str, list[Any]]]:
    """level -> (state_fips, state) for the levels that are states; the others by kind.

    In a table that also lists countries (two or more levels that are neither states nor parts of
    one), a state name that is also a country's ("Georgia") is ambiguous and left out, named."""
    rows: list[dict[str, Any]] = []
    others: dict[str, list[Any]] = {}
    foreign = sum(geography.classify(level, codes=True).kind == "unknown" for level in levels)
    for level in levels:
        if foreign >= 2 and str(level).strip().casefold() in geography.COUNTRY_NAMESAKES:
            others.setdefault("ambiguous", []).append(level)
            continue
        place = geography.classify(level, codes=True)  # the state role is already established
        if place.kind == "state" and place.state is not None:
            rows.append({"__level": str(level), _STATE_FIPS: place.state.fips, _STATE_NAME: place.state.name})
        elif place.kind == "nation":
            rows.append({"__level": str(level), _STATE_FIPS: "US", _STATE_NAME: "United States"})
            others.setdefault("nation", []).append(level)
        else:
            others.setdefault(place.kind, []).append(level)
    schema = {"__level": pl.String, _STATE_FIPS: pl.String, _STATE_NAME: pl.String}
    return pl.DataFrame(rows, schema=schema), others


def _count_expr(source: SourceInput, value: ValueColumn) -> pl.Expr:
    """True on rows whose value is a count, fail closed (``estimates.count_row_expr``)."""
    table = source.table
    if table is None:
        return pl.lit(True)
    return estimates.count_row_expr(value, table)


def _harmonize(
    source: SourceInput, value: ValueColumn, pins: _Pins, asked: Asked, count: bool
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """The table's rows with harmonized keys and its value columns, plus notes.

    Raises:
        JoinRefusedError: a role the question needs can't be applied to this table.
    """
    notes: list[str] = []
    table = source.table
    assert table is not None  # noqa: S101 - callers refused undetected tables
    # Withheld over the whole table, before any filter: a row with a small count anywhere in it, or
    # a total with a withheld part, is never shown (estimates.withheld_mask).
    lazy = source.frame.with_columns(estimates.withheld_mask(source.frame, table).alias("__withheld")).lazy()
    for column, level in pins.pinned.items():
        lazy = lazy.filter(pl.col(column) == level)
    for column, levels in pins.kept.items():
        lazy = lazy.filter(pl.col(column).is_in(levels))
    keys: list[str] = []
    exprs: list[pl.Expr] = []
    # state
    state_col = pins.roles.get("state")
    if state_col is not None:
        levels = source.frame.get_column(state_col).drop_nulls().unique().to_list()
        mapping, others = _state_map(levels)
        lazy = (
            lazy.with_columns(pl.col(state_col).cast(pl.String).alias("__level"))
            .join(mapping.lazy(), on="__level", how="left")
            .drop("__level")
        )
        for level in others.get("ambiguous", []):
            notes.append(config.ESTIMATE_JOIN_AMBIGUOUS_PLACE.format(source=source.name, level=level))
        substate = others.get("substate", []) + others.get("unknown", [])
        if substate:
            notes.append(
                config.ESTIMATE_JOIN_NOT_STATES.format(
                    source=source.name,
                    levels=", ".join(str(v) for v in substate[: config.ESTIMATE_LEVELS_IN_HOLD]),
                    count=len(substate),
                )
            )
        keys += [_STATE_FIPS, _STATE_NAME]
    elif "state" in pins.constants:
        column, level = pins.constants["state"]
        place = geography.classify(level, geography.is_geo_code_column(column))
        if place.kind == "state" and place.state is not None:
            exprs += [pl.lit(place.state.fips).alias(_STATE_FIPS), pl.lit(place.state.name).alias(_STATE_NAME)]
            keys += [_STATE_FIPS, _STATE_NAME]
    # year / season
    year_col = pins.roles.get("year")
    season_col = pins.roles.get("season")
    if pins.year_aligned is not None:
        exprs.append(pl.lit(pins.year_aligned, dtype=pl.Int64).alias("year"))
        keys.append("year")
    elif year_col is not None:
        exprs.append(pl.col(year_col).cast(pl.Int64, strict=False).alias("year"))
        keys.append("year")
    elif season_col is not None:
        raise JoinRefusedError(config.ESTIMATE_JOIN_SEASON_TREND.format(source=source.name, column=season_col))
    elif "year" in pins.constants:
        exprs.append(pl.lit(int(pins.constants["year"][1]), dtype=pl.Int64).alias("year"))
        keys.append("year")
    # sex
    sex_col = pins.roles.get("sex")
    if sex_col is not None:
        sex_map = {
            str(level): _sex(level) for level in source.frame.get_column(sex_col).drop_nulls().unique().to_list()
        }
        exprs.append(pl.col(sex_col).cast(pl.String).replace_strict(sex_map, default=None).alias("sex"))
        keys.append("sex")
    age_col = pins.roles.get("age")
    if age_col is not None:
        exprs.append(pl.col(age_col).cast(pl.String).alias("age"))
        keys.append("age")
    for column in pins.groups:
        exprs.append(pl.col(column).alias(column))
        keys.append(column)
    exprs += [
        pl.col(value.name).cast(pl.Float64, strict=False).alias("_v"),
        (pl.col(value.lower).cast(pl.Float64, strict=False) if value.lower else pl.lit(None, pl.Float64)).alias("_lo"),
        (pl.col(value.upper).cast(pl.Float64, strict=False) if value.upper else pl.lit(None, pl.Float64)).alias("_hi"),
        (pl.col(value.suppressed).fill_null(False) if value.suppressed else pl.lit(False)).alias("_sup"),
        (pl.col(value.unreliable).fill_null(False) if value.unreliable else pl.lit(False)).alias("_unr"),
        _count_expr(source, value).alias("_cnt"),
        pl.col("__withheld").alias("_wh"),
    ]
    frame = lazy.with_columns(exprs).select([*keys, *_VALUE_COLUMNS]).collect()
    count = count or bool(frame.get_column("_cnt").any())
    # Filters from the question's roles.
    before = frame.height
    null_keys = frame.filter(pl.any_horizontal([pl.col(k).is_null() for k in keys])) if keys else frame.clear()
    if null_keys.height:
        notes.append(config.ESTIMATE_JOIN_EMPTY_KEYS.format(source=source.name, count=null_keys.height))
        frame = frame.drop_nulls(subset=keys)
    if _STATE_FIPS in keys:
        # A ranking or a screen compares a named state with the others: it keeps every state (or
        # the named group's), and the named state is the one the answer is about.
        among = asked.analysis in ("rank", "disagree", "precision")
        if among and asked.groups:
            frame = frame.filter(pl.col(_STATE_FIPS).is_in(asked.states))
        elif among:
            nation = frame.filter(pl.col(_STATE_FIPS) == "US").height
            frame = frame.filter(pl.col(_STATE_FIPS) != "US")
            if nation:
                notes.append(config.ESTIMATE_JOIN_NATION_LEFT_OUT.format(source=source.name))
        elif asked.states:
            wanted = list(asked.states) + (["US"] if asked.nation else [])
            frame = frame.filter(pl.col(_STATE_FIPS).is_in(wanted))
        elif asked.nation and not asked.all_states:
            frame = frame.filter(pl.col(_STATE_FIPS) == "US")
        else:
            nation = frame.filter(pl.col(_STATE_FIPS) == "US").height
            frame = frame.filter(pl.col(_STATE_FIPS) != "US")
            if nation:
                notes.append(config.ESTIMATE_JOIN_NATION_LEFT_OUT.format(source=source.name))
    if "year" in keys and asked.years and pins.year_aligned is None:
        if asked.trend and len(asked.years) >= 2:
            frame = frame.filter(pl.col("year").is_between(min(asked.years), max(asked.years)))
        else:
            frame = frame.filter(pl.col("year").is_in(asked.years))
    if "sex" in keys:
        if asked.sexes:
            frame = frame.filter(pl.col("sex").is_in(asked.sexes))
        elif asked.by_sex:
            frame = frame.filter(pl.col("sex") != "Both")
        if count and frame.get_column("sex").n_unique() > 1 and "Both" in frame.get_column("sex").to_list():
            # A count total beside its parts would give away a hidden part by subtraction.
            frame = frame.filter(pl.col("sex") != "Both")
            notes.append(config.ESTIMATE_JOIN_COUNT_TOTAL_DROPPED.format(source=source.name))
    if "age" in keys and age_col is not None:
        dimension = source.table.dimension(age_col) if source.table else None
        summaries = list(dimension.parents) if dimension else []
        if summaries:
            # Summary ages (All ages, Age-standardized, a band containing others) are never a group.
            frame = frame.filter(~pl.col("age").is_in([str(v) for v in summaries]))
            notes.append(config.ESTIMATE_JOIN_AGE_SUMMARIES_DROPPED.format(source=source.name))
    logger.info("estimate_join_harmonized", dataset_id=source.dataset_id, rows_in=before, rows_kept=frame.height)
    return frame, keys, notes


# ---------------------------------------------------------------------------- prepare


def _resolve_roles(source: SourceInput, pins: _Pins, asked: Asked, choices: Mapping[str, Any]) -> None:
    """Holds and refusals for the join roles of one table (state, year, season, sex)."""
    table = source.table
    assert table is not None  # noqa: S101
    year_col = pins.roles.get("year") or pins.roles.get("season")
    if year_col is not None:
        dimension = table.dimension(year_col)
        assert dimension is not None  # noqa: S101
        available = [v for v in dimension.values if v is not None]
        # A year or season choice pairs one table's year with the asked year: it holds for this
        # request only, never saved and reused for another year.
        chosen = choices.get(year_col)
        season = pins.roles.get("season") == year_col
        if len(asked.years) == 1 and not asked.trend:
            year = asked.years[0]
            if chosen is not None and chosen in available and (season or chosen != year):
                pins.pinned[year_col] = chosen
                pins.year_aligned = year
                pins.roles.pop("season" if season else "year", None)
            elif season:
                default = next((s for s in available if _season_end(s) == year), None) or (
                    available[-1] if available else None
                )
                pins.holds.append(
                    (
                        config.ESTIMATE_JOIN_SEASON_HOLD.format(
                            source=source.name, column=year_col, year=year, default=default
                        ),
                        HoldOption(source.dataset_id, source.name, year_col, default, tuple(available)),
                    )
                )
            elif year not in available and str(year) not in [str(v) for v in available]:
                earlier = [v for v in available if isinstance(v, int) and v < year]
                default = max(earlier) if earlier else (min(available) if available else None)
                pins.holds.append(
                    (
                        config.ESTIMATE_JOIN_YEAR_MISSING.format(
                            source=source.name, year=year, first=min(available), last=max(available), default=default
                        ),
                        HoldOption(source.dataset_id, source.name, year_col, default, tuple(available)),
                    )
                )
        elif season:
            pins.refusals.append(config.ESTIMATE_JOIN_SEASON_TREND.format(source=source.name, column=year_col))
        elif not asked.years and not asked.trend and asked.analysis in ("table", "rank", "disagree", "precision"):
            pins.refusals.append(
                config.ESTIMATE_JOIN_NAME_YEAR.format(source=source.name, first=min(available), last=max(available))
            )
    elif "year" in pins.constants and len(asked.years) == 1 and not asked.trend:
        column, constant = pins.constants["year"]
        if int(constant) != asked.years[0]:
            if choices.get(column) == constant:
                pins.year_aligned = asked.years[0]
            else:
                pins.holds.append(
                    (
                        config.ESTIMATE_JOIN_YEAR_MISSING.format(
                            source=source.name, year=asked.years[0], first=constant, last=constant, default=constant
                        ),
                        HoldOption(source.dataset_id, source.name, column, constant, (constant,)),
                    )
                )
    sex_col = pins.roles.get("sex")
    if sex_col is None:
        if asked.by_sex or any(s != "Both" for s in asked.sexes):
            pins.refusals.append(config.ESTIMATE_JOIN_NO_SEX.format(source=source.name))
    elif not asked.sexes and not asked.by_sex:
        dimension = table.dimension(sex_col)
        assert dimension is not None  # noqa: S101
        chosen = _choice(source, choices, sex_col)
        both = next((v for v in dimension.values if _sex(v) == "Both"), None)
        if chosen is not None and chosen in dimension.values:
            pins.pinned[sex_col] = chosen
        elif both is not None:
            pins.holds.append(
                (
                    config.ESTIMATE_JOIN_OPEN_HOLD.format(
                        source=source.name, column=sex_col, levels=_levels_text(dimension), top=both
                    ),
                    HoldOption(source.dataset_id, source.name, sex_col, both, dimension.values),
                )
            )
        else:
            pins.refusals.append(
                config.ESTIMATE_JOIN_OPEN_REFUSAL.format(
                    source=source.name, column=sex_col, levels=_levels_text(dimension)
                )
            )
    state_col = pins.roles.get("state")
    if state_col is None and "state" not in pins.constants and (asked.states or asked.analysis != "table"):
        pins.refusals.append(config.ESTIMATE_JOIN_NO_STATE.format(source=source.name))
    if (
        state_col is not None
        and not asked.states
        and not asked.all_states
        and not asked.nation
        and asked.analysis in ("table", "change")
    ):
        pins.refusals.append(config.ESTIMATE_JOIN_NAME_STATE.format(source=source.name))


def _season_end(level: Any) -> int | None:
    match = _SEASON.match(str(level))
    if match is None:
        return None
    start, end = match.groups()
    return int(end) if len(end) == 4 else int(start[:2] + end)


def _wayne_trap(sources: Sequence[SourceInput], asked: Asked) -> str | None:
    """A question about Wayne County on a table whose Wayne rows exclude Detroit (out-Wayne)."""
    if not geography.mentions_wayne(asked.text):
        return None
    for source in sources:
        if source.table is not None:
            found = estimates.out_wayne_refusal(source.table, asked.text, source.name)
            if found is not None:
                return found
    return config.ESTIMATE_JOIN_COUNTY_GRAIN


def prepare(
    sources: Sequence[SourceInput], question: str, choices: Mapping[str, Mapping[str, Any]]
) -> tuple[Asked, list[Prepared] | JoinAnswer]:
    """Every table pinned and harmonized for the question, or the hold/refusal that stops it."""
    asked = read_question(question)
    for source in sources:
        if not source.declared:
            return asked, _refused(config.ESTIMATE_JOIN_UNDECLARED.format(source=source.name))
        if source.table is None:
            return asked, _refused(config.ESTIMATE_JOIN_NOT_ESTIMATES.format(source=source.name))
    wayne = _wayne_trap(sources, asked)
    if wayne is not None:
        return asked, _refused(wayne)
    for source in sources:
        county = _county_column(source)
        if county is not None:
            return asked, _refused(config.ESTIMATE_JOIN_COUNTY_TABLE.format(source=source.name, column=county))
    if asked.analysis == "disagree" and len(sources) != 2:
        return asked, _refused(config.ESTIMATE_JOIN_DISAGREE_TWO)
    if asked.analysis == "precision" and len(sources) != 1:
        return asked, _refused(config.ESTIMATE_JOIN_PRECISION_ONE)
    single = len(sources) == 1
    holds: list[tuple[str, HoldOption]] = []
    refusals: list[str] = []
    pinned_all: list[tuple[SourceInput, _Pins, ValueColumn]] = []
    sources_used = list(sources)
    by_source = standardization_by_source(sources, asked.text)
    asked_for = {s.dataset_id: dataclasses.replace(asked, standardization=by_source[s.dataset_id]) for s in sources}
    question_asked = asked
    for index, source in enumerate(sources):
        own = choices.get(source.dataset_id, {})
        asked = asked_for[source.dataset_id]
        # A choice made now is read like one saved earlier (the request's wins).
        source = dataclasses.replace(source, saved_choices={**source.saved_choices, **own})
        sources_used[index] = source
        pins = _pin_dimensions(source, asked, own, single)
        _resolve_roles(source, pins, asked, own)
        value, value_hold = _choose_value(source, asked, own)
        if value_hold is not None:
            holds.append(
                (
                    config.ESTIMATE_JOIN_WHICH_VALUE.format(source=source.name, values=", ".join(value_hold.levels)),
                    value_hold,
                )
            )
        holds += pins.holds
        refusals += pins.refusals
        if value is not None:
            pinned_all.append((source, pins, value))
    if refusals:
        return asked, _refused(" ".join(dict.fromkeys(refusals)))
    if holds:
        return asked, JoinAnswer(
            status="held",
            message=" ".join(dict.fromkeys(text for text, _ in holds)) + " " + config.ESTIMATE_JOIN_CONFIRM,
            options=[option for _, option in holds],
        )
    prepared: list[Prepared] = []
    for source, pins, value in pinned_all:
        fixed = {**_constant_attributes(source), **{c: v for c, v in pins.constants.values()}, **pins.pinned}
        measure = measure_of(source, value, fixed)
        # Standardization matters where values are compared across tables, not side by side.
        if measure.kind == "rate" and measure.standardization == "unknown" and asked.analysis == "disagree":
            option = HoldOption(source.dataset_id, source.name, STANDARDIZATION_CHOICE, "crude", STANDARDIZATION_LEVELS)
            return asked, JoinAnswer(
                status="held",
                message=config.ESTIMATE_JOIN_STANDARDIZATION_UNKNOWN.format(source=source.name, value=value.name),
                options=[option],
            )
        try:
            frame, keys, notes = _harmonize(source, value, pins, asked_for[source.dataset_id], measure.kind == "count")
        except JoinRefusedError as e:
            return asked, _refused(str(e))
        duplicated = frame.group_by(keys).len().filter(pl.col("len") > 1).height if keys else int(frame.height > 1)
        if duplicated:
            logger.error("estimate_join_grain_violated", dataset_id=source.dataset_id, keys=keys, groups=duplicated)
            return asked, _refused(
                config.ESTIMATE_JOIN_GRAIN.format(source=source.name, keys=", ".join(keys) or "none")
            )
        pinned_text = ", ".join(f"{k} = {v}" for k, v in fixed.items())
        aligned = (
            config.ESTIMATE_JOIN_YEAR_ALIGNED.format(year=pins.year_aligned) if pins.year_aligned is not None else ""
        )
        method = config.ESTIMATE_JOIN_METHOD_SOURCE.format(
            source=source.name,
            value=value.name,
            bounds=f" with {value.lower}–{value.upper}"
            if value.lower and value.upper
            else " (no interval in the file)",
            pinned=f" for {pinned_text}" if pinned_text else "",
            measure=_describe_measure(measure),
            rows=frame.height,
            aligned=aligned,
        )
        prepared.append(
            Prepared(source, value, measure, frame, keys, list(pins.groups), dict(pins.pinned), method, notes)
        )
    if all(p.frame.height == 0 for p in prepared):
        return question_asked, _refused(config.ESTIMATE_NO_ROWS)
    return question_asked, prepared


def _constant_attributes(source: SourceInput) -> dict[str, Any]:
    """Text and key columns with one value in the whole table (measure_name = Deaths): what the
    file is about, named in the method so the reader sees what was read. Detection files such
    columns as aliases of one another, so they are read from the rows, not the key."""
    table = source.table
    assert table is not None  # noqa: S101
    measured = {c for v in table.values for c in (v.name, v.lower, v.upper, v.suppressed, v.unreliable) if c}
    keys = {d.name for d in table.dimensions}
    candidates = [
        c
        for c, dtype in source.frame.schema.items()
        if c not in measured and c not in keys and (dtype == pl.String or dtype.is_integer())
    ]
    if not candidates:
        return {}
    counts = source.frame.select(pl.col(candidates).drop_nulls().n_unique()).row(0, named=True)
    single = [c for c in candidates if counts[c] == 1]
    if not single:
        return {}
    firsts = source.frame.select(pl.col(single).drop_nulls().first()).row(0, named=True)
    return {c: firsts[c] for c in single}


def _county_column(source: SourceInput) -> str | None:
    """The column that makes ``source`` a county (or city) table with no state key, or None.
    Its levels, or those of a column that repeats it (a five-digit FIPS), are mostly parts of
    states."""
    table = source.table
    assert table is not None  # noqa: S101
    if any(role_of(d) == "state" for d in table.dimensions):
        return None
    for dimension in table.dimensions:
        for column in (dimension.name, *dimension.aliases):
            levels = source.frame.get_column(column).drop_nulls().unique().head(config.ESTIMATE_MAX_LISTED_LEVELS)
            codes = geography.is_geo_code_column(column)
            kinds = [geography.classify(v, codes).kind for v in levels.to_list()]
            if kinds and kinds.count("substate") * 2 >= len(kinds):
                return dimension.name
    return None


def _refused(message: str) -> JoinAnswer:
    return JoinAnswer(status="refused", message=message)


# ---------------------------------------------------------------------------- join


def _join(prepared: Sequence[Prepared]) -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
    """Inner join on the keys the tables share; returns (joined, keys, joined_on, notes)."""
    notes: list[str] = []
    joined_on: list[str] = []
    first = prepared[0]
    joined = first.frame.rename({c: f"{c}0" for c in _VALUE_COLUMNS})
    keys = list(first.keys)
    for index, part in enumerate(prepared[1:], start=1):
        renamed = part.frame.rename({c: f"{c}{index}" for c in _VALUE_COLUMNS})
        shared = [k for k in keys if k in part.keys and k != _STATE_NAME]
        if not shared:
            raise JoinRefusedError(config.ESTIMATE_JOIN_NO_SHARED_KEY.format(a=first.source.name, b=part.source.name))
        right = renamed
        if _STATE_NAME in right.columns and _STATE_NAME in joined.columns:
            right = right.drop(_STATE_NAME)
        only_left = joined.join(right.select(shared), on=shared, how="anti").height
        only_right = right.join(joined.select(shared), on=shared, how="anti").height
        joined = joined.join(right, on=shared, how="inner")
        joined_on += [k for k in shared if k not in joined_on]
        keys += [k for k in part.keys if k not in keys]
        if only_left or only_right:
            notes.append(
                config.ESTIMATE_JOIN_UNMATCHED.format(
                    a=first.source.name if index == 1 else "the tables before",
                    a_only=only_left,
                    b=part.source.name,
                    b_only=only_right,
                )
            )
    for part in prepared:
        # A key the table doesn't have but the joined rows vary on would repeat its one value
        # beside every level (a state-only value shown as if it were each year's): refused.
        missing = [k for k in keys if k not in part.keys and k != _STATE_NAME and joined.get_column(k).n_unique() > 1]
        if missing:
            raise JoinRefusedError(
                config.ESTIMATE_JOIN_FAN_OUT.format(source=part.source.name, keys=", ".join(missing))
            )
    return joined, keys, joined_on, notes


# ---------------------------------------------------------------------------- formatting


def _fmt(value: float | None) -> str:
    return estimates._fmt(value)


def _cell(
    row: Mapping[str, Any], index: int, measure: Measure
) -> tuple[float | None, float | None, float | None, str | None]:
    """(value, lower, upper, hidden reason) for one table's value on one joined row."""
    value, lower, upper = row[f"_v{index}"], row[f"_lo{index}"], row[f"_hi{index}"]
    if row[f"_sup{index}"]:
        return None, None, None, "source"
    if (measure.kind == "count" or row[f"_cnt{index}"]) and estimates.small_count(value):
        return None, None, None, "small"
    if row[f"_wh{index}"]:
        return None, None, None, "withheld"
    if value is not None and lower is not None and upper is not None and not lower <= value <= upper:
        return value, None, None, "interval"
    return value, lower, upper, None


def _label(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    shown = [k for k in keys if k != _STATE_FIPS]
    return " · ".join(str(row[k]) for k in shown if row.get(k) is not None)


def _value_text(value: float | None, lower: float | None, upper: float | None, hidden: str | None) -> str:
    if hidden in _HIDDEN:
        return SECONDARY_SUPPRESSED
    return _fmt(value) + estimates._interval(lower, upper)


def _hidden_notes(counts: Mapping[str, int], unreliable: Sequence[str]) -> list[str]:
    notes = []
    if counts.get("small"):
        notes.append(config.ESTIMATE_SMALL_COUNT_NOTE)
    if counts.get("source"):
        notes.append(config.ESTIMATE_SOURCE_SUPPRESSED_NOTE)
    if counts.get("withheld"):
        notes.append(config.ESTIMATE_WITHHELD_NOTE)
    if counts.get("interval"):
        notes.append(config.ESTIMATE_BAD_INTERVAL_NOTE.format(groups=f"{counts['interval']} rows"))
    if unreliable:
        notes.append(
            config.ESTIMATE_UNRELIABLE_NOTE.format(groups=", ".join(unreliable[: config.ESTIMATE_LEVELS_IN_HOLD]))
        )
    return notes


# ---------------------------------------------------------------------------- analyses


def _table(
    prepared: Sequence[Prepared], joined: pl.DataFrame, keys: Sequence[str]
) -> tuple[str, dict[str, Any], list[str]]:
    hidden: dict[str, int] = {}
    unreliable: list[str] = []
    columns = [k for k in keys if k != _STATE_FIPS]
    rows: list[dict[str, Any]] = []
    sort_keys = [k for k in (_STATE_NAME, "year", "sex", "age") if k in keys] + [
        k for k in keys if k not in _KEYS and k not in (_STATE_FIPS, _STATE_NAME)
    ]
    for row in joined.sort(sort_keys).iter_rows(named=True):
        shown = {k: row[k] for k in columns}
        for index, part in enumerate(prepared):
            value, lower, upper, why = _cell(row, index, part.measure)
            if why:
                hidden[why] = hidden.get(why, 0) + 1
            if row[f"_unr{index}"] and why is None:
                unreliable.append(_label(row, keys))
            name = f"{part.source.name}: {part.value.name}"
            shown[name] = SECONDARY_SUPPRESSED if why in _HIDDEN else value
            if part.value.lower and part.value.upper:
                shown[f"{name} lower"] = SECONDARY_SUPPRESSED if why in _HIDDEN else lower
                shown[f"{name} upper"] = SECONDARY_SUPPRESSED if why in _HIDDEN else upper
        rows.append(shown)
    names = " and ".join(p.source.name for p in prepared)
    if len(rows) == 1:
        row = joined.row(0, named=True)
        parts = []
        for index, part in enumerate(prepared):
            parts.append(f"{part.source.name} {part.value.name} {_value_text(*_cell(row, index, part.measure))}")
        claim = f"{_label(row, keys)}: " + "; ".join(parts) + "."
    else:
        claim = config.ESTIMATE_JOIN_TABLE_CLAIM.format(rows=len(rows), sources=names)
    return claim, {"columns": list(rows[0]) if rows else [], "rows": rows}, _hidden_notes(hidden, unreliable)


def _change(
    prepared: Sequence[Prepared], joined: pl.DataFrame, keys: Sequence[str]
) -> tuple[str, dict[str, Any], list[str]]:
    if "year" not in keys:
        raise JoinRefusedError(config.ESTIMATE_JOIN_CHANGE_NO_YEAR)
    years = sorted(joined.get_column("year").unique().to_list())
    if len(years) < 2:
        raise JoinRefusedError(config.ESTIMATE_JOIN_CHANGE_ONE_YEAR.format(years=", ".join(map(str, years)) or "none"))
    group_keys = [k for k in keys if k not in ("year", _STATE_FIPS)]
    claim_parts: list[str] = []
    hidden: dict[str, int] = {}
    change_rows: list[dict[str, Any]] = []
    first_year, last_year = years[0], years[-1]
    by_group = joined.partition_by(group_keys, as_dict=True, maintain_order=True) if group_keys else {(): joined}
    for group, frame in by_group.items():
        label = " · ".join(str(g) for g in group) if group_keys else "all"
        start = frame.filter(pl.col("year") == first_year)
        end = frame.filter(pl.col("year") == last_year)
        if start.height != 1 or end.height != 1:
            continue
        a, b = start.row(0, named=True), end.row(0, named=True)
        for index, part in enumerate(prepared):
            va, la, ua, wa = _cell(a, index, part.measure)
            vb, lb, ub, wb = _cell(b, index, part.measure)
            entry: dict[str, Any] = {
                "group": label,
                "table": part.source.name,
                f"{first_year}": None,
                f"{last_year}": None,
            }
            if wa in _HIDDEN or wb in _HIDDEN or va is None or vb is None:
                for why in (wa, wb):
                    if why:
                        hidden[why] = hidden.get(why, 0) + 1
                entry |= {
                    f"{first_year}": SECONDARY_SUPPRESSED,
                    f"{last_year}": SECONDARY_SUPPRESSED,
                    "change": SECONDARY_SUPPRESSED,
                    "percent change": SECONDARY_SUPPRESSED,
                }
                change_rows.append(entry)
                claim_parts.append(config.ESTIMATE_JOIN_CHANGE_HIDDEN.format(group=label, source=part.source.name))
                continue
            difference = vb - va
            percent = (difference / va * 100) if va else None
            overlap = estimates._overlap({"lower": la, "upper": ua}, {"lower": lb, "upper": ub})
            verdict = (
                config.ESTIMATE_JOIN_CHANGE_BAD_INTERVAL
                if "interval" in (wa, wb)
                else config.ESTIMATE_JOIN_CHANGE_NO_BOUNDS
                if overlap is None
                else config.ESTIMATE_JOIN_CHANGE_OVERLAP
                if overlap
                else config.ESTIMATE_JOIN_CHANGE_APART
            )
            entry |= {
                f"{first_year}": va,
                f"{first_year} interval": estimates._interval(la, ua).strip(" ()") or None,
                f"{last_year}": vb,
                f"{last_year} interval": estimates._interval(lb, ub).strip(" ()") or None,
                "change": round(difference, 4),
                "percent change": round(percent, 1) if percent is not None else None,
            }
            change_rows.append(entry)
            pct = f", {percent:+.1f}%" if percent is not None else ""
            claim_parts.append(
                f"{label} ({part.source.name}): {_fmt(va)}{estimates._interval(la, ua)} in {first_year} → "
                f"{_fmt(vb)}{estimates._interval(lb, ub)} in {last_year}{pct}. {verdict}"
            )
    if not change_rows:
        raise JoinRefusedError(config.ESTIMATE_JOIN_CHANGE_ONE_YEAR.format(years=", ".join(map(str, years))))
    shown_claims = claim_parts[: config.ESTIMATE_LEVELS_IN_HOLD]
    more = len(claim_parts) - len(shown_claims)
    claim = " ".join(shown_claims) + (f" ({more} more in the table.)" if more > 0 else "")
    notes = [config.ESTIMATE_JOIN_CHANGE_METHOD, *_hidden_notes(hidden, [])]
    if len(years) > 2:
        trend_claim, trend_table, trend_notes = _table(prepared, joined, keys)
        return claim, {"columns": list(change_rows[0]), "rows": change_rows, "years": trend_table}, notes + trend_notes
    return claim, {"columns": list(change_rows[0]), "rows": change_rows}, notes


def _rank_target(prepared: Sequence[Prepared], asked: Asked) -> int:
    if len(prepared) == 1:
        return 0
    named = [i for i, p in enumerate(prepared) if estimates._phrase_in(p.source.name, asked.words)]
    if len(named) == 1:
        return named[0]
    raise JoinRefusedError(config.ESTIMATE_JOIN_RANK_WHICH.format(sources=", ".join(p.source.name for p in prepared)))


def rank_range(
    value: float,
    lower: float | None,
    upper: float | None,
    others: Sequence[tuple[float, float | None, float | None]],
    lowest_first: bool,
) -> tuple[int, int, int]:
    """(point rank, best rank, worst rank) of one estimate among ``others``, highest first unless
    ``lowest_first``. The range counts states whose interval lies entirely beyond this one's as
    certainly ahead and states whose interval overlaps it as possibly ahead."""
    lo = lower if lower is not None else value
    hi = upper if upper is not None else value
    sign = -1 if lowest_first else 1
    point = 1 + sum(1 for v, _l, _u in others if sign * v > sign * value)
    certain = possible = 0
    for v, l_other, u_other in others:
        ol = l_other if l_other is not None else v
        ou = u_other if u_other is not None else v
        if lowest_first:
            certain += ou < lo
            possible += ol <= hi
        else:
            certain += ol > hi
            possible += ou >= lo
    return point, 1 + certain, 1 + possible


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rank(
    prepared: Sequence[Prepared], joined: pl.DataFrame, keys: Sequence[str], asked: Asked
) -> tuple[str, dict[str, Any], list[str]]:
    if _STATE_FIPS not in keys:
        raise JoinRefusedError(config.ESTIMATE_JOIN_NO_STATE.format(source=prepared[0].source.name))
    target = _rank_target(prepared, asked)
    part = prepared[target]
    others_keys = [k for k in keys if k not in (_STATE_FIPS, _STATE_NAME)]
    varying = [k for k in others_keys if joined.get_column(k).n_unique() > 1]
    if varying:
        raise JoinRefusedError(config.ESTIMATE_JOIN_RANK_OPEN.format(columns=", ".join(varying)))
    hidden: dict[str, int] = {}
    entries = []
    for row in joined.iter_rows(named=True):
        value, lower, upper, why = _cell(row, target, part.measure)
        if why in _HIDDEN or value is None:
            hidden[why or "missing"] = hidden.get(why or "missing", 0) + 1
            continue
        entries.append((row[_STATE_NAME], row[_STATE_FIPS], value, lower, upper))
    if not entries:
        raise JoinRefusedError(config.ESTIMATE_NO_ROWS)
    sign = 1 if asked.lowest_first else -1
    entries.sort(key=lambda e: sign * e[2])
    rows: list[dict[str, Any]] = []
    ranges: dict[str, tuple[int, int, int]] = {}
    for name, fips, value, lower, upper in entries:
        others = [(v, lo, hi) for n, _f, v, lo, hi in entries if n != name]
        ranges[fips] = rank_range(value, lower, upper, others, asked.lowest_first)
        point, best, worst = ranges[fips]
        rows.append(
            {
                "rank": point,
                "state": name,
                part.value.name: value,
                "lower": lower,
                "upper": upper,
                "rank range": f"{best}–{worst}" if lower is not None and upper is not None else None,
            }
        )
    total = len(entries)
    order = "lowest first" if asked.lowest_first else "highest first"
    focus = [e for e in entries if e[1] in asked.focus]
    scope = ", ".join(f"{k} = {joined.get_column(k)[0]}" for k in others_keys)
    scope = f" ({scope})" if scope else ""
    if len(focus) == 1:
        name, fips, value, lower, upper = focus[0]
        point, best, worst = ranges[fips]
        claim = config.ESTIMATE_JOIN_RANK_FOCUS.format(
            state=name,
            rank=_ordinal(point),
            total=total,
            order=order,
            source=part.source.name,
            value=part.value.name,
            shown=_fmt(value) + estimates._interval(lower, upper),
            scope=scope,
        )
        if lower is not None and upper is not None:
            claim += " " + (
                config.ESTIMATE_JOIN_RANK_CERTAIN
                if best == worst
                else config.ESTIMATE_JOIN_RANK_RANGE.format(best=_ordinal(best), worst=_ordinal(worst))
            )
        else:
            claim += " " + config.ESTIMATE_JOIN_RANK_NO_BOUNDS
    else:
        top = entries[: min(5, total)]
        listed = "; ".join(
            f"{_ordinal(ranges[f][0])} {n} {_fmt(v)}{estimates._interval(lo, hi)}" for n, f, v, lo, hi in top
        )
        claim = config.ESTIMATE_JOIN_RANK_TOP.format(
            total=total, order=order, source=part.source.name, value=part.value.name, scope=scope, listed=listed
        )
    notes = [config.ESTIMATE_JOIN_RANK_METHOD, *_hidden_notes(hidden, [])]
    if hidden.get("small") or hidden.get("source"):
        notes.append(config.ESTIMATE_JOIN_RANK_HIDDEN.format(count=hidden.get("small", 0) + hidden.get("source", 0)))
    return claim, {"columns": list(rows[0]), "rows": rows}, notes


def _disagree(
    prepared: Sequence[Prepared], joined: pl.DataFrame, keys: Sequence[str]
) -> tuple[str, dict[str, Any], list[str]]:
    a, b = prepared
    reason = comparable(a, b)
    if reason is not None:
        raise JoinRefusedError(reason)
    a_bounds = bool(a.value.lower and a.value.upper)
    b_bounds = bool(b.value.lower and b.value.upper)
    if not (a_bounds or b_bounds):
        raise JoinRefusedError(config.ESTIMATE_JOIN_DISAGREE_NO_BOUNDS.format(a=a.source.name, b=b.source.name))
    model, observed = (0, 1) if a_bounds else (1, 0)
    m, o = prepared[model], prepared[observed]
    both = a_bounds and b_bounds
    hidden: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    above = below = inside = 0
    unreliable: list[str] = []
    gaps: list[tuple[float, str, str]] = []
    for row in joined.iter_rows(named=True):
        vm, lm, um, wm = _cell(row, model, m.measure)
        vo, lo, uo, wo = _cell(row, observed, o.measure)
        label = _label(row, keys)
        if wm in _HIDDEN or wo in _HIDDEN or vm is None or vo is None:
            for why in (wm, wo):
                if why:
                    hidden[why] = hidden.get(why, 0) + 1
            continue
        if row[f"_unr{observed}"] or row[f"_unr{model}"]:
            unreliable.append(label)
        if lm is None or um is None:
            verdict = "no interval"
        elif both and lo is not None and uo is not None:
            verdict = "intervals overlap" if lo <= um and lm <= uo else ("above" if lo > um else "below")
        else:
            verdict = "above" if vo > um else "below" if vo < lm else "inside"
        above += verdict == "above"
        below += verdict == "below"
        inside += verdict in ("inside", "intervals overlap")
        ratio = vo / vm if vm else None
        if ratio is not None and verdict in ("above", "below"):
            gaps.append((ratio, label, verdict))
        rows.append(
            {
                "group": label,
                f"{o.source.name}: {o.value.name}": vo,
                f"{m.source.name}: {m.value.name}": vm,
                f"{m.source.name} lower": lm,
                f"{m.source.name} upper": um,
                "reported vs interval": verdict,
                "ratio": round(ratio, 3) if ratio is not None else None,
            }
        )
    if not rows:
        raise JoinRefusedError(config.ESTIMATE_NO_ROWS)
    rows.sort(key=lambda r: -(r["ratio"] or 0.0))
    gaps.sort(key=lambda g: -abs(_log_ratio(g[0])))
    listed = ", ".join(f"{label} {ratio:.2f}×" for ratio, label, _ in gaps[:3]) or "none"
    claim = config.ESTIMATE_JOIN_DISAGREE_CLAIM.format(
        total=len(rows),
        observed=o.source.name,
        model=m.source.name,
        above=above,
        below=below,
        inside=inside,
        listed=listed,
        test="overlaps" if both else "falls inside",
    )
    notes = [
        config.ESTIMATE_JOIN_DISAGREE_METHOD.format(observed=o.source.name, model=m.source.name),
        *_hidden_notes(hidden, unreliable),
    ]
    return claim, {"columns": list(rows[0]), "rows": rows}, notes


def _log_ratio(ratio: float) -> float:
    return math.log(ratio) if ratio > 0 else float("inf")


def _precision(
    prepared: Sequence[Prepared], joined: pl.DataFrame, keys: Sequence[str], asked: Asked
) -> tuple[str, dict[str, Any], list[str]]:
    part = prepared[0]
    if not (part.value.lower and part.value.upper):
        raise JoinRefusedError(config.ESTIMATE_JOIN_PRECISION_NO_BOUNDS.format(source=part.source.name))
    group_keys = [k for k in keys if k != _STATE_FIPS]
    varying = [k for k in group_keys if k not in (_STATE_NAME, *part.groups) and joined.get_column(k).n_unique() > 1]
    if varying:
        raise JoinRefusedError(config.ESTIMATE_JOIN_RANK_OPEN.format(columns=", ".join(varying)))
    hidden: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    skipped = bad = 0
    for row in joined.iter_rows(named=True):
        value, lower, upper, why = _cell(row, 0, part.measure)
        if why in _HIDDEN:
            hidden[why] = hidden.get(why, 0) + 1
            continue
        if why == "interval":
            bad += 1
            continue
        if value is None or lower is None or upper is None or value <= 0:
            skipped += 1
            continue
        label = " · ".join(str(row[k]) for k in (_STATE_NAME, *part.groups) if k in row)
        rows.append(
            {
                "group": label,
                part.value.name: value,
                "lower": lower,
                "upper": upper,
                "relative width": round((upper - lower) / value, 3),
            }
        )
    if not rows:
        raise JoinRefusedError(config.ESTIMATE_NO_ROWS)
    rows.sort(key=lambda r: -r["relative width"])
    listed = "; ".join(
        f"{r['group']} {r['relative width']:.2f} "
        f"({_fmt(r['lower'])} to {_fmt(r['upper'])} around {_fmt(r[part.value.name])})"
        for r in rows[:5]
    )
    claim = config.ESTIMATE_JOIN_PRECISION_CLAIM.format(total=len(rows), source=part.source.name, listed=listed)
    notes = [config.ESTIMATE_JOIN_PRECISION_METHOD, *_hidden_notes(hidden, [])]
    if skipped:
        notes.append(config.ESTIMATE_JOIN_PRECISION_SKIPPED.format(count=skipped))
    if bad:
        notes.append(config.ESTIMATE_JOIN_PRECISION_BAD.format(count=bad))
    return claim, {"columns": list(rows[0]), "rows": rows}, notes


# ---------------------------------------------------------------------------- entry point


def savable_choices(source: SourceInput, choices: Mapping[str, Any]) -> dict[str, Any]:
    """The request's choices worth saving for later questions on ``source``: everything except a
    year or season pairing, which is only right for the year it was confirmed with."""
    table = source.table
    if table is None:
        return {}
    kept: dict[str, Any] = {}
    for column, value in choices.items():
        dimension = table.dimension(column)
        if dimension is not None and role_of(dimension) in ("year", "season"):
            continue
        if dimension is None and column not in (VALUE_CHOICE, STANDARDIZATION_CHOICE):
            continue  # a one-value column (a constant year) chosen for one pairing
        kept[column] = value
    return kept


def choice_problem(source: SourceInput, choices: Mapping[str, Any]) -> str | None:
    """Why ``choices`` can't apply to ``source`` (an unknown column or level), or None."""
    table = source.table
    for column, value in choices.items():
        if column == STANDARDIZATION_CHOICE:
            if value not in STANDARDIZATION_LEVELS:
                return f"{STANDARDIZATION_CHOICE} must be one of {', '.join(STANDARDIZATION_LEVELS)}."
            continue
        if table is None:
            return f"{source.name} is not a declared estimate table, so no choice applies to it."
        if column == VALUE_CHOICE:
            if not any(v.name == value for v in table.values):
                return f"{source.name} has no value column {value!r}."
            continue
        dimension = table.dimension(column)
        if dimension is not None:
            if dimension.values and value not in dimension.values:
                return f"{source.name}: {value!r} is not a value of {column}."
            continue
        if column in table.attributes and value in source.frame.get_column(column).unique().to_list():
            continue
        return f"{source.name} has no column {column!r} to choose a value for."
    return None


def answer(
    sources: Sequence[SourceInput], question: str, choices: Mapping[str, Mapping[str, Any]] | None = None
) -> JoinAnswer:
    """Answer, hold or refuse a question across ``sources`` (one or more declared estimate tables)."""
    log = logger.bind(sources=len(sources))
    if not sources:
        return _refused(config.ESTIMATE_JOIN_NO_SOURCES)
    asked, prepared = prepare(sources, question, choices or {})
    if isinstance(prepared, JoinAnswer):
        log.info("estimate_join_stopped", status=prepared.status, analysis=asked.analysis)
        return prepared
    try:
        if len(prepared) > 1:
            joined, keys, joined_on, join_notes = _join(prepared)
        else:
            joined = prepared[0].frame.rename({c: f"{c}0" for c in _VALUE_COLUMNS})
            keys, joined_on, join_notes = prepared[0].keys, [], []
        if joined.height == 0:
            raise JoinRefusedError(config.ESTIMATE_JOIN_NOTHING_MATCHED)
        if asked.analysis == "change":
            claim, table, notes = _change(prepared, joined, keys)
        elif asked.analysis == "rank":
            claim, table, notes = _rank(prepared, joined, keys, asked)
        elif asked.analysis == "disagree":
            claim, table, notes = _disagree(prepared, joined, keys)
        elif asked.analysis == "precision":
            claim, table, notes = _precision(prepared, joined, keys, asked)
        else:
            claim, table, notes = _table(prepared, joined, keys)
            if _has(asked.text, config.ESTIMATE_JOIN_POOL_WORDS):
                claim = config.ESTIMATE_JOIN_NO_POOL + " " + claim
    except JoinRefusedError as e:
        log.info("estimate_join_refused", analysis=asked.analysis)
        return _refused(str(e))
    method = [p.method for p in prepared]
    if len(prepared) > 1:
        method.append(config.ESTIMATE_JOIN_METHOD_JOIN.format(keys=" × ".join(joined_on), rows=joined.height))
    source_notes = [n for p in prepared for n in p.notes]
    log.info("estimate_join_answered", analysis=asked.analysis, rows=joined.height)
    return JoinAnswer(
        status="answered",
        message=claim,
        table=table,
        method=method,
        notes=list(dict.fromkeys([*notes, *join_notes, *source_notes])),
    )
