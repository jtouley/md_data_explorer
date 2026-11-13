# SPDX-License-Identifier: MIT
"""Mixed units in a question (DQ M3): no statistic, limit or group over two scales.

A column the question uses (metric, limit, group, share, predictor, cohort step, numbers stored
as text) is read whole, before any of the question's limits, and checked with
``contracts.unit_mix.detect_unit_mix``. A mix is held, and the hold offers what works wherever
the column is used:

* convert: values on one side of a round cut point are read in the other unit and converted
  (a numeric column whose groups a cut point separates). Stored as a ``UnitRule``;
* one unit: every value is in one unit ("children beside adults, all kg"): the header's unit
  when it names one, else either unit, when every value is plausible in it. Stored as a
  ``OneUnitDeclaration``; the hold is lifted and nothing is converted.

The doctor picks one (``UnitChoice``); a plain confirmation does not pick for them. When
neither is possible the question is refused: only the file can be corrected.

Once a column is converted, a number the question compares it with ("glucose above 7") is in an
unknown unit. It is converted when the question names its unit ("above 7 mmol/L"), used as it is
when it names the column's unit, and otherwise held until the doctor says which unit it is in.

Sentences name columns, units, round cut points and the question's own numbers, never a
patient's value or a count.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from clinical_analytics.analysis.variable_types import short_name
from clinical_analytics.contracts import config
from clinical_analytics.contracts.declarations import OneUnitDeclaration, UnitRule
from clinical_analytics.contracts.unit_mix import UnitMix, detect_unit_mix, relation_text, unit_pair_for
from clinical_analytics.core.answer_checks import plan_columns, value_counts
from clinical_analytics.core.plan_values import plan_values
from clinical_analytics.core.query_plan import QueryPlan

# "low"/"high": every value is in the pair's low/high unit (a OneUnitDeclaration).
UnitsChoice: TypeAlias = Literal["convert", "low", "high"]
LimitsChoice: TypeAlias = Literal["target", "source"]


@dataclass(frozen=True)
class UnitChoice:
    """What the doctor chose on a unit hold: convert the mix or declare it one unit (``units``),
    and whether the question's limits are in the column's unit or the other (``limits``)."""

    units: UnitsChoice | None = None
    limits: LimitsChoice | None = None

    @property
    def key(self) -> str:
        return f"units={self.units or ''};limits={self.limits or ''}"


# Every choice a hold can offer, so a confirmation's hash is matched before the checks run.
UNIT_CHOICES: tuple[UnitChoice, ...] = tuple(
    UnitChoice(units, limits)
    for units in (None, "convert", "low", "high")
    for limits in (None, "target", "source")
    if units or limits
)


@dataclass(frozen=True)
class UnitOption:
    """One way to settle a unit hold: the label the doctor reads, and what choosing it does."""

    label: str
    choice: UnitChoice


@dataclass(frozen=True)
class MixedColumn:
    """A plan column read as a mix of two units, and what the doctor may confirm about it."""

    mix: UnitMix
    rule: UnitRule | None  # the conversion (None: no cut point, or numbers stored as text)
    one_units: tuple[OneUnitDeclaration, ...]  # "every value is in X", for each unit X that can be

    def one_unit(self, units: UnitsChoice | None) -> OneUnitDeclaration | None:
        """The declaration for choice ``units`` ("low"/"high"), if this column offers it."""
        pair = self.mix.pair
        unit = pair.low.unit if units == "low" else pair.high.unit if units == "high" else None
        return next((d for d in self.one_units if d.unit == unit), None)


@dataclass(frozen=True)
class LimitUnit:
    """A number the question compares a converted column with. ``stated``: the unit the question
    names right after the number, None when it names none."""

    column: str
    value: float
    stated: str | None
    rule: UnitRule

    @property
    def converted(self) -> float:
        """The number read in ``rule.source_unit`` and converted to the column's unit."""
        return self.value * self.rule.scale + self.rule.offset


def mixed_columns(
    plan: QueryPlan, layer: Any, dataset_version: str | None, predictors: list[str], declared: dict[str, str]
) -> list[MixedColumn]:
    """Each plan column (and predictor) that mixes two units, read whole, unless the doctor
    declared it one unit on this version (``declared``). The unit pair and column type are read
    once per column and passed down."""
    view = layer.get_base_view()
    found = []
    for column in dict.fromkeys([*plan_columns(plan), *predictors]):
        if column in declared or column not in view.columns:
            continue
        pair = unit_pair_for(column)
        if pair is None:
            continue
        dtype = view[column].type()
        text = bool(dtype.is_string())
        if not (text or (dtype.is_numeric() and not dtype.is_boolean())):
            continue
        values, counts = value_counts(layer, column)
        mix = detect_unit_mix(column, values, counts, pair=pair)
        if mix is None:
            continue
        rule = None if text else mix.rule(dataset_version)
        ones = tuple(
            OneUnitDeclaration(column=column, unit=unit, dataset_version=dataset_version) for unit in mix.one_units
        )
        found.append(MixedColumn(mix=mix, rule=rule, one_units=ones))
    return found


def _side_text(side: str, cutoff: float) -> str:
    return f"below {cutoff:g}" if side == "below" else f"of {cutoff:g} and above"


def _why(mix: UnitMix) -> str:
    """Why the column looks mixed, so the doctor can tell two units from two kinds of patient."""
    relation = relation_text(mix.pair)
    if mix.reason == "implausible":
        return f"some values are impossible in {mix.target} but plausible in {mix.other} ({relation})"
    if mix.reason == "spread":
        return f"its values spread as widely as a mix of the two units would ({relation})"
    return f"its values form two groups as far apart as the two units ({relation})"


def mix_sentence(found: MixedColumn) -> str:
    """The hold (or, with nothing to confirm, the refusal) for a column that mixes two units."""
    mix, rule = found.mix, found.rule
    name = short_name(mix.column)
    head = f"{name} looks like it mixes {mix.target} and {mix.other}: {_why(mix)}."
    fix = "correct the file so each value is in one unit"
    one = (
        f"confirm that every value is in one unit ({' or '.join(d.unit for d in found.one_units)}), "
        "if the two groups are two kinds of patient"
        if found.one_units
        else ""
    )
    if rule is not None:
        convert = f"convert values {_side_text(rule.side, rule.cutoff)} from {rule.source_unit} to {rule.target_unit}"
        return f"{head} Choose to {convert}, or {one}." if one else f"{head} Choose to {convert}, or {fix}."
    why_not = (
        "The two groups overlap, so no cut point separates them."
        if mix.cutoff is None
        else "Values stored as text are not converted."
    )
    if one:
        return f"{head} {why_not} {one[0].upper()}{one[1:]}, or {fix}."
    return f"{head} {why_not} Some values are impossible in either unit alone, so {fix}."


def settleable(found: MixedColumn) -> bool:
    return found.rule is not None or bool(found.one_units)


def _unit_key(text: str) -> str:
    """'mmol/L' -> 'mmoll', '°F' -> 'f', 'µmol/L' -> 'umoll' (normalised like header suffixes)."""
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("µ", "u").replace("μ", "u"))


def _unit_names(pair: config.UnitPair) -> dict[str, str]:
    """Normalised unit spelling -> unit, for both units of ``pair``."""
    names: dict[str, str] = {}
    for bounds, hints in ((pair.low, pair.low_hints), (pair.high, pair.high_hints)):
        for spelling in (_unit_key(bounds.unit), *hints):
            names[spelling] = bounds.unit
    return names


# A number and the word right after it: "7 mmol/L", "126mg/dl", "100.4 °F", "154 lbs".
_NUMBER_UNIT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(°\s*[a-z]+|[a-zµμ][a-z0-9µμ]*(?:\s*/\s*[a-z]+)?)", re.I)


def stated_unit(question: str, value: float, pair: config.UnitPair) -> str | None:
    """The unit of ``pair`` the question writes right after ``value``, or None."""
    names = _unit_names(pair)
    for number, word in _NUMBER_UNIT.findall(question):
        if math.isclose(float(number), value, rel_tol=1e-9, abs_tol=1e-12) and _unit_key(word) in names:
            return names[_unit_key(word)]
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def unit_limits(plan: QueryPlan, question: str, rules: dict[str, UnitRule]) -> list[LimitUnit]:
    """Every number the plan compares a converted column with (``rules``: column -> the rule its
    values went through last), with the unit the question states for it."""
    found = []
    for column, value, _compared in plan_values(plan):
        rule, number = rules.get(column), _number(value)
        pair = unit_pair_for(column)
        if rule is None or number is None or pair is None:
            continue
        found.append(LimitUnit(column=column, value=number, stated=stated_unit(question, number, pair), rule=rule))
    return list(dict.fromkeys(found))


def unstated(limits: Iterable[LimitUnit]) -> list[LimitUnit]:
    """Limits whose unit is not known: the question names none, or a unit of neither side."""
    return [limit for limit in limits if limit.stated not in (limit.rule.source_unit, limit.rule.target_unit)]


def limit_sentence(limits: list[LimitUnit]) -> str:
    """The hold for limits with no unit on a converted column."""
    named = ", ".join(f"{limit.value:g} for {short_name(limit.column)}" for limit in limits)
    rule = limits[0].rule
    return (
        f"The limit {named} does not say its unit, and {short_name(rule.column)} is read in {rule.target_unit} "
        f"after converting {rule.source_unit} values. Choose which unit the limit is in."
    )


def unit_options(
    mixed: list[MixedColumn], stored_limits: list[LimitUnit], pending_limits: list[LimitUnit]
) -> list[UnitOption]:
    """The ways to settle the question's unit holds, each one complete. ``stored_limits``: limits
    with no unit on columns already converted; ``pending_limits``: on columns a conversion now
    offered would convert."""
    unit_steps: list[tuple[UnitsChoice | None, str]] = []
    if mixed:
        if all(found.rule is not None for found in mixed):
            unit_steps.append(
                (
                    "convert",
                    "; ".join(
                        f"Convert {short_name(f.mix.column)} values {_side_text(f.rule.side, f.rule.cutoff)} "
                        f"from {f.rule.source_unit} to {f.rule.target_unit}"
                        for f in mixed
                        if f.rule is not None
                    ),
                )
            )
        sides: list[UnitsChoice] = ["low", "high"]
        if mixed[0].one_units and mixed[0].one_units[0].unit == mixed[0].mix.pair.high.unit:
            sides.reverse()  # the header's (or larger group's) unit first
        for side in sides:
            declared = [found.one_unit(side) for found in mixed]
            if all(d is not None for d in declared):
                unit_steps.append(
                    (side, "; ".join(f"Every {short_name(d.column)} value is in {d.unit}" for d in declared if d))
                )
    else:
        unit_steps.append((None, ""))
    options = []
    for units, label in unit_steps:
        limits = [*stored_limits, *(pending_limits if units == "convert" else [])]
        if not limits:
            if units is not None:
                options.append(UnitOption(label=label, choice=UnitChoice(units=units)))
            continue
        in_unit: LimitsChoice
        for in_unit in ("target", "source"):
            limit_label = _limit_label(limits, in_unit)
            text = f"{label}; {limit_label}" if label else limit_label
            options.append(UnitOption(label=text, choice=UnitChoice(units=units, limits=in_unit)))
    return options


def _limit_label(limits: list[LimitUnit], side: LimitsChoice) -> str:
    parts = []
    for limit in limits:
        rule = limit.rule
        if side == "target":
            parts.append(f"{limit.value:g} is in {rule.target_unit}")
        else:
            parts.append(f"{limit.value:g} is in {rule.source_unit} ({limit.converted:g} {rule.target_unit})")
    return "; ".join(f"Limit {part}" for part in parts)


def converted_plan(plan: QueryPlan, conversions: list[LimitUnit]) -> QueryPlan:
    """``plan`` with each limit in ``conversions`` read in its column's unit (7 mmol/L -> 126.1)."""
    if not conversions:
        return plan
    lookup = {(limit.column, limit.value): limit.converted for limit in conversions}

    def mapped(column: str | None, value: Any) -> Any:
        if isinstance(value, list):
            return [mapped(column, item) for item in value]
        number = _number(value)
        return lookup.get((str(column), number), value) if number is not None else value

    return dataclasses.replace(
        plan,
        filters=[dataclasses.replace(f, value=mapped(f.column, f.value)) for f in plan.filters],
        cohort_steps=[dataclasses.replace(s, value=mapped(s.column, s.value)) for s in plan.cohort_steps],
        proportion_value=mapped(plan.proportion_column, plan.proportion_value),
        rate_value=mapped(plan.rate_column, plan.rate_value),
    )


def limit_line_note(limit: LimitUnit) -> str:
    """'limit 7 mmol/L on glucose read as 126.112 mg/dL' for the plan line."""
    rule = limit.rule
    return (
        f"limit {limit.value:g} {rule.source_unit} on {short_name(limit.column)} read as "
        f"{limit.converted:g} {rule.target_unit}"
    )


def unit_rule_text(rule: UnitRule) -> str:
    """'glucose values below 30 read as mmol/L and converted to mg/dL (1 mmol/L = 18.016 mg/dL)'."""
    side = _side_text(rule.side, rule.cutoff)
    return (
        f"{short_name(rule.column)} values {side} read as {rule.source_unit} and converted to "
        f"{rule.target_unit} ({rule.relation})"
    )


def one_unit_line(column: str, unit: str) -> str:
    """'every weight value read as kg' (a one-unit declaration, as the plan line names it)."""
    return f"every {short_name(column)} value read as {unit}"


def one_unit_note(column: str, unit: str) -> str:
    return f"Data check: {one_unit_line(column, unit)}, as confirmed."


def applied_unit_rules(plan: QueryPlan, rules: tuple[UnitRule, ...], predictors: list[str]) -> list[UnitRule]:
    """The confirmed unit rules on the columns this plan reads, in the order confirmed."""
    columns = {*plan_columns(plan), *predictors}
    return [rule for rule in rules if rule.column in columns]


def last_rules(rules: Iterable[UnitRule]) -> dict[str, UnitRule]:
    """Column -> the last rule its values went through (the unit it is read in now)."""
    return {rule.column: rule for rule in rules}


def with_unit_rules(plan_line: str, rules: list[UnitRule], notes: list[str] | None = None) -> str:
    """The plan line with each applied unit conversion (and converted limit) named, so the line
    says what ran."""
    parts = [unit_rule_text(rule) for rule in rules] + list(notes or [])
    if not parts:
        return plan_line
    return f"{plan_line} · units {'; '.join(parts)}"
