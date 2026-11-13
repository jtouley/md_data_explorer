# SPDX-License-Identifier: MIT
"""Bind the unmatched tail of a question to columns and their stored values.

Cohort limits are FilterSpec-shaped steps. The only operators added here that a
plain equality filter cannot express are a median cutoff and a 90th-percentile
cutoff. Column choice comes from token overlap with the column name and with
the values actually stored in it, including the code labels in the header.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from clinical_analytics.core.column_parser import parse_column_name
from clinical_analytics.core.query_plan import STATISTIC_WORDS, CohortStep, Statistic
from clinical_analytics.core.question_vocabulary import (
    INVERSE_OP,
    NEGATION_CUES,
    NO_LIMIT_WORDS,
    RANGE_PATTERN,
    SUBJECT_WORDS,
    NegationScope,
    negation_scope,
    postfix_negated,
    range_ends,
    range_in_order,
)

_OVER = re.compile(r"\b(?:over|above|older than)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
# Comparisons: "<direction> N" and "N <direction>" ("65+", "65 or older", "at least 65").
_PRE_COMPARE = re.compile(
    r"\b(?P<dir>over|above|exceeding|under|below|older than|younger than|more than|greater than|higher than|"
    r"less than|fewer than|lower than|at least|at most)\s+(?P<num>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_POST_COMPARE = re.compile(
    r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<dir>\+|plus\b|(?:or|and) (?:more|older|over|above|greater|higher|less|fewer|"
    r"younger|under|below|lower)\b(?! than)(?!\s+\d))",
    re.IGNORECASE,
)
_PRE_OPS = {"at least": "ge", "at most": "le"}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_LOWER_WORDS = ("under", "below", "younger", "less", "fewer", "lower")
_DECILE = re.compile(
    r"\b(?:top|highest|oldest) decile(?: of ([a-z0-9 ]+?))?(?=[?]|,|$)",
    re.IGNORECASE,
)
_MEDIAN_CUT = re.compile(
    r"\b(?:older|greater|higher) than the (?:cohort )?median(?: ([a-z0-9 ]+?))?(?=[?]|,|$)",
    re.IGNORECASE,
)
# "compare <metric> by|between|across <group>": the group is what follows the preposition.
_COMPARE_SPLIT = re.compile(
    r"\bcompare\s+(?P<metric>.+?)\s+(?:by|between|across)\s+(?P<group>.+?)\s*[?.]?$",
    re.IGNORECASE,
)
# Trailing "by <group>": that column is the grouping, never a filter on itself.
_BY_GROUP = re.compile(r"\bby\s+(?P<group>[a-z0-9 :]+?)\s*[?.]?$", re.IGNORECASE)
# "in each site", "per site": the same grouping as "by site".
_EACH_GROUP = re.compile(r"\b(?:each|per)\s+(?P<group>[a-z0-9 :]+?)\s*[?.]?$", re.IGNORECASE)
# Words that end the measure named after a statistic word and start a limit or a grouping.
_METRIC_END = re.compile(
    r"\b(?:in|for|who|with|on|among|by|within|of|from|at|over|under|above|below|aged|older|younger|between|"
    r"excluding|except|outside|not|without)\b"
)
# "proportion of <denominator> are/who/have ... <numerator>": the words before the verb name
# the population, the words after it the thing being counted.
_SHARE_SPLIT = re.compile(
    r"\b(?:proportion|share|percentage|percent|fraction)\s+of\s+(?P<den>.+?)\s+"
    r"(?:are|is|who|that|have|has|had|were|was|use|uses|using|on|with|taking|take|takes)\b",
    re.IGNORECASE,
)
_SEX_WORDS = {"sex", "gender", "woman", "women", "man", "men", "female", "male"}
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "and",
    "or",
    "by",
    "with",
    "who",
    "that",
    "those",
    "patient",
    "patients",
    "how",
    "many",
    "what",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "average",
    "mean",
    "median",
    "among",
    "between",
    "clinical",
    "question",
    "compare",
    "correlation",
    "proportion",
    "share",
    "more",
    "likely",
    "than",
    "their",
    "this",
    "these",
    "not",
    "no",
    "yes",
    "stayed",
    "admitted",
    "prescribed",
}

# English inflections of a value that is already stored on the column.
_VALUE_INFLECTIONS = {
    "female": {"woman", "women"},
    "male": {"man", "men"},
    "f": {"female", "woman", "women"},
    "m": {"male", "man", "men"},
}


@dataclass
class PeelResult:
    """Cohort restrictions removed from a question, plus the text left to parse."""

    residual: str
    steps: list[CohortStep] = field(default_factory=list)
    proportion_column: str | None = None
    proportion_value: str | int | float | None = None
    rate_column: str | None = None
    rate_value: str | int | float | None = None
    locked_metric: str | None = None
    locked_group: str | None = None
    force_intent: str | None = None
    statistic: Statistic | None = None


def asked_statistic(text: str, median_is_cutoff: bool = False) -> Statistic | None:
    """The statistic the question's first statistic word asks for ("median age" -> median).

    "median" in a cutoff ("older than the median age") limits the patients; it is not the
    statistic asked for, so it is skipped when ``median_is_cutoff``.
    """
    for word in re.findall(r"[a-z]+", text.lower()):
        if word == "median" and median_is_cutoff:
            continue
        if word in STATISTIC_WORDS:
            return STATISTIC_WORDS[word]
    return None


def peel_cohort(
    query: str,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
) -> PeelResult:
    """Split cohort constraints from the measurable part of a question."""
    text = " ".join(query.split())
    lowered = text.lower()
    consumed: set[str] = set()
    hidden: list[str] = []
    steps: list[tuple[int, CohortStep]] = []

    age = _best_column(columns, {"age"})
    ranged: set[int] = set()
    for match in RANGE_PATTERN.finditer(lowered):
        # Both ends belong to the column the words before the lower end name. Unknown owner,
        # a negation anywhere before it, or a reversed range ("between 65 and 50"): leave it
        # unbound, so the plan holds.
        _low, _high, lo_group, hi_group = range_ends(match)
        ranged |= {match.start(lo_group), match.start(hi_group)}
        if _scope(lowered, match.start()) != "clear" or not range_in_order(match):
            continue
        tokens_before = _owner_tokens(lowered[: match.end(lo_group)])
        named = _owner_of(tokens_before, columns, values_for)
        if named:
            low, high = float(match.group(lo_group)), float(match.group(hi_group))
            steps.append((0, CohortStep("ge", named, low)))
            steps.append((0, CohortStep("le", named, high)))
            consumed |= _tokens(named)
            hidden.append(match.group(0))
    for match, op in _comparisons(lowered):
        if match.start("num") in ranged:
            continue
        # The number belongs to the column the question names before it ("bmi over 30");
        # age only when the words say so. Unknown owner: leave it unbound, so the plan holds
        # instead of guessing. "not over 65" is "65 or under". A negation that may or may not
        # reach the comparison ("excluding those with ldl over 160"): never bind it as asked.
        scope = _scope(lowered, match.start())
        if scope == "uncertain":
            continue
        if scope == "negated":
            op = INVERSE_OP[op]
        named = _number_owner(lowered, match, columns, values_for)
        if named:
            steps.append((0, CohortStep(op, named, float(match.group("num")))))  # type: ignore[arg-type]
            consumed |= _tokens(named) | _tokens(match.group(0))
            hidden.append(match.group(0))

    decile = _DECILE.search(text)
    if decile:
        named = _best_column(columns, _tokens(decile.group(1) or "")) or age
        if named:
            steps.append((1, CohortStep("ge_percent_rank", named, quantile=0.9)))
            consumed |= _tokens(named) | {"decile", "top", "highest", "oldest"}
            hidden.append(decile.group(0))

    median_cut = _MEDIAN_CUT.search(text)
    if median_cut:
        named = _best_column(columns, _tokens(median_cut.group(1) or "")) or age
        if named:
            steps.append((1, CohortStep("gt_median", named)))
            consumed |= _tokens(named) | {"median", "higher", "greater", "older"}
            hidden.append(median_cut.group(0))

    if re.search(r"\bcorrelat", lowered):
        ranked = _ranked(columns, _tokens(lowered))
        found = [column for column, score in ranked if score > 0][:2]
        if len(found) >= 2:
            return PeelResult(
                residual="what is the correlation?",
                steps=[step for _phase, step in steps],
                locked_metric=found[0],
                locked_group=found[1],
                force_intent="CORRELATIONS",
            )

    value_hits = _value_hits(lowered, columns, values_for)
    name_tokens = set().union(*(_tokens(column) for column in columns)) if columns else set()
    comparing = re.search(r"\bcompare\b|\bby sex\b|\bby gender\b", lowered) is not None
    # Sex words are the comparison's groups unless it names another grouping column
    # ("compare ldl by treatment in women": women is a limit).
    compare_split = _COMPARE_SPLIT.search(lowered) if comparing else None
    compare_by = _best_column(columns, _tokens(compare_split.group("group"))) if compare_split else None
    sex_is_group = comparing and (compare_by is None or _is_group(compare_by))
    rating = re.search(r"\b(proportion|share|percentage|percent|fraction|more likely)\b", lowered) is not None
    more_likely = re.search(r"\bmore likely\b", lowered) is not None
    # "are <group> more likely to <rate>": the words before "more likely" name the groups
    # compared, the words after it the thing whose rate is compared. Never the other way.
    rate_at = lowered.find("more likely") if more_likely else None
    rate_groups: set[str] = set()
    # Share questions: denominator before the verb, numerator after it. Without that
    # structure the direction is unknown, so no proportion is set and the plan holds.
    share_split = _SHARE_SPLIT.search(lowered) if rating and not more_likely else None
    numerator_at = share_split.end("den") if share_split else None
    numerators: list[tuple[str, Any]] = []
    counting = re.search(r"\bhow many\b", lowered) is not None or rating
    mentioned = _mentioned(lowered, columns, _tokens(lowered) - consumed)
    bound_columns = {step.column for _phase, step in steps}
    by_group = _BY_GROUP.search(lowered) or _EACH_GROUP.search(lowered)
    if by_group:
        grouping_column = _best_column(columns, _tokens(by_group.group("group")))
        if grouping_column:
            bound_columns.add(grouping_column)

    for column, value, at in value_hits:
        scope = _scope(lowered, at)
        if scope == "uncertain":
            # "without diabetes and hypertension", "excluding women on placebo": the negation
            # may cover this value too. Unbound, so the plan holds.
            continue
        negated = scope == "negated"
        if negated and (isinstance(value, list) or (numerator_at is not None and at >= numerator_at)):
            # "not in the north or west", "share ... not on placebo": no single exclusion
            # step says that, so the value stays unbound and the plan holds.
            continue
        if isinstance(value, list) and (rate_at is not None or (numerator_at is not None and at >= numerator_at)):
            # "more likely to be in the north or west", "share ... in the north or west": a rate or
            # share of several values at once is not a plan this layer computes, and the rate's
            # own column must never become a cohort limit. Unbound, so the plan holds.
            continue
        if rate_at is not None and (at > rate_at or not negated):
            if at < rate_at:
                rate_groups.add(column)
                bound_columns.add(column)
            # A value after "more likely" is not a rate this plan can compute: leave it unbound.
            continue
        if (
            column in bound_columns
            or (sex_is_group and _is_group(column))
            or (re.search(r"\bmore likely\b", lowered) and _is_group(column))
        ):
            continue
        labels = value if isinstance(value, list) else [value]
        # Digits count: "Type 2" is named by "type 2", not by the column word "type" alone.
        overlap = set().union(*(_tokens(str(label), digits=True) for label in labels)) & _tokens(lowered, digits=True)
        foreign = name_tokens - _tokens(column)
        if overlap and overlap <= foreign:
            continue
        if overlap and overlap <= _tokens(column):
            # "most common Current Regimen" names the column; "regimen" inside the
            # stored value "regimen_1" is not a request to filter to that value.
            continue
        if numerator_at is not None and at >= numerator_at:
            numerators.append((column, value))
            bound_columns.add(column)
            continue
        if negated:
            # "not on placebo": everyone with another recorded value. Never == placebo.
            steps.append((0, CohortStep("ne", column, value)))
        elif isinstance(value, list):
            # "in the north or west": any of the named values.
            steps.append((0, CohortStep("in", column, value)))
        else:
            steps.append((0, CohortStep("eq", column, value)))
        bound_columns.add(column)
        for label in labels:
            consumed |= _tokens(str(label)) | _VALUE_INFLECTIONS.get(str(label).strip().lower(), set())

    binary_hits = []
    for column, at in mentioned:
        if column in bound_columns or _qualified(lowered, at, columns, values_for):
            continue
        polarity = _binary_polarity(column, values_for(column), lowered, at)
        if polarity is None:
            temporal = _temporal_value(values_for(column))
            if temporal is not None and re.search(r"\b(have|has|recorded|non-null)\b", lowered):
                steps.append((2 if counting else 0, CohortStep("not_null", column)))
                bound_columns.add(column)
            continue
        if polarity[0] == "ne" and (rating or numerator_at is not None):
            # A share or rate of "any value but none" is not one stored value: leave it unbound.
            continue
        binary_hits.append((at, column, polarity))

    binary_hits.sort(key=lambda item: item[0])
    if rate_at is not None:
        rate_groups |= {column for at, column, _polarity in binary_hits if at < rate_at}
        binary_hits = [hit for hit in binary_hits if hit[0] > rate_at]
    outcome = binary_hits[-1][1] if binary_hits and (counting or rating) else None
    proportion_column = None
    proportion_value = None
    rate_column = None
    rate_value = None
    for at, column, (op, polarity) in binary_hits:
        if numerator_at is not None:
            if at >= numerator_at:
                numerators.append((column, polarity))
            else:
                steps.append((0, CohortStep(op, column, polarity)))  # type: ignore[arg-type]
            bound_columns.add(column)
            continue
        phase = 2 if column == outcome else 0
        if rating and more_likely and column == outcome:
            rate_column, rate_value = column, polarity
            bound_columns.add(column)
            continue
        steps.append((phase, CohortStep(op, column, polarity)))  # type: ignore[arg-type]
        bound_columns.add(column)
    if len(numerators) == 1 and not isinstance(numerators[0][1], list):
        # One numerator value only: a share of an "or" list is never a plan (left unbound above).
        proportion_column, proportion_value = numerators[0]

    locked_metric = None
    locked_group = None
    force_intent = None
    regrouped: str | None = None
    if comparing:
        # The group comes from the words after by/between/across. Sex is only the default
        # when the question itself is about sex; never substitute it for a named group.
        split = _COMPARE_SPLIT.search(lowered)
        if split:
            group = _best_column(columns, _tokens(split.group("group")))
            metric_tokens = _tokens(split.group("metric")) - consumed
            if group is None:
                # "between women and men", "between the north and the south": the group is
                # the column whose values are named; naming some of them limits to those.
                group, named_values = _group_from_values(split.group("group"), columns, values_for)
                if group is not None:
                    # Parse the comparison in its plain form; the limit travels as a step.
                    regrouped = f"compare {split.group('metric')} by {group}"
                if group is not None and named_values is not None:
                    steps.append((0, CohortStep("in", group, named_values)))
                    bound_columns.add(group)
        else:
            group = _best_column(columns, {"sex", "gender"}) if _tokens(lowered) & _SEX_WORDS else None
            metric_tokens = _tokens(lowered) - consumed
        if group:
            metric_tokens -= _tokens(group)
        metric = _best_column(
            [column for column in columns if column not in bound_columns],
            metric_tokens,
        )
        if metric:
            locked_metric = metric
        if group:
            locked_group = group
        force_intent = "COMPARE_GROUPS"
    elif _EACH_GROUP.search(lowered) and not rating:
        each = _EACH_GROUP.search(lowered)
        locked_group = _best_column(columns, _tokens(each.group("group"))) if each else None
    elif rate_column and rate_at is not None:
        # The groups are what the words before "more likely" name; sex only when they say so.
        if len(rate_groups) == 1:
            locked_group = next(iter(rate_groups))
        elif not rate_groups and _tokens(lowered[:rate_at]) & _SEX_WORDS:
            locked_group = _best_column(columns, {"sex", "gender"})

    ordered = [step for _phase, step in sorted(steps, key=lambda item: item[0])]
    keep: set[str] = set()
    stat_phrase = re.search(
        r"\b(?:average|mean|median|minimum|maximum|min|max|std)\s+([a-z0-9 ]+)",
        lowered,
    )
    if stat_phrase and locked_metric is None:
        # The measure is the words right after the statistic word, up to the first word that
        # starts a limit ("average ldl within the south site", "average ldl of diabetics").
        segments = re.split(_METRIC_END, stat_phrase.group(1))
        phrase = next((segment for segment in segments if _tokens(segment)), "")
        chosen = _best_column(columns, _tokens(phrase))
        if chosen:
            keep = _tokens(chosen)
            locked_metric = chosen
    visible = text
    for span in hidden:
        visible = re.sub(re.escape(span), " ", visible, count=1, flags=re.IGNORECASE)
    dropped = (consumed | _bound_tokens(ordered, columns)) - keep
    # Nothing peeled: the downstream tiers must see the question exactly as asked.
    residual = text if not hidden and not dropped else _cleanup(_drop_tokens(visible, dropped))
    if regrouped is not None:
        residual = regrouped
    if rate_column or proportion_column:
        residual = "how many patients?"
    return PeelResult(
        residual=residual,
        steps=ordered,
        statistic=asked_statistic(lowered, median_is_cutoff=median_cut is not None),
        proportion_column=proportion_column,
        proportion_value=proportion_value,
        rate_column=rate_column,
        rate_value=rate_value,
        locked_metric=locked_metric,
        locked_group=locked_group,
        force_intent=force_intent,
    )


def _group_from_values(
    phrase: str, columns: list[str], values_for: Callable[[str], list[Any]]
) -> tuple[str | None, list[Any] | None]:
    """(column, values) when the words name two or more whole values of exactly one column.

    The values are None when every stored value is named (the grouping covers them all).
    Anything else, including values of two columns, is (None, None): no group is guessed.
    """
    tokens = _tokens(phrase, digits=True)
    found: list[tuple[str, list[Any], int]] = []
    for column in columns:
        stored = [value for value in values_for(column) if value is not None and not isinstance(value, bool)]
        named = [
            value
            for value in stored
            if not _NUMBER.fullmatch(str(value).strip())
            and (_label_at(phrase, tokens, str(value).strip().lower()) or (3, 0))[0] <= 1
        ]
        if len(named) >= 2:
            found.append((column, named, len(stored)))
    if len(found) != 1:
        return None, None
    column, named, stored_count = found[0]
    return column, (None if len(named) == stored_count else named)


def attach_cohort(intent: Any, peeled: PeelResult) -> Any:
    """Copy peeled restrictions onto the parsed intent.

    Confidence is never raised: the result keeps the parsed intent's confidence. When the
    residual did not parse, an intent is built at confidence 0.0, so the execution gate
    holds it for confirmation instead of running it.
    """
    if intent is not None and peeled.statistic:
        # The asked statistic travels with the plan, so execution computes it and word
        # accounting can check it; a plan without it leaves the statistic word unused.
        intent.statistic = peeled.statistic
    # A grouping from "in each site" is structure too: dropping it would answer the total.
    has_structure = bool(
        peeled.steps or peeled.proportion_column or peeled.rate_column or peeled.force_intent or peeled.locked_group
    )
    if not has_structure:
        return intent
    if intent is None:
        from clinical_analytics.core.nl_query_engine import QueryIntent

        intent = QueryIntent(
            intent_type=peeled.force_intent or "COUNT",
            confidence=0.0,
            parsing_tier="pattern_match",
        )
    intent.cohort_steps.extend(peeled.steps)
    if peeled.statistic:
        intent.statistic = peeled.statistic
    if peeled.proportion_column:
        intent.proportion_column = peeled.proportion_column
        intent.proportion_value = peeled.proportion_value
    if peeled.rate_column:
        intent.rate_column = peeled.rate_column
        intent.rate_value = peeled.rate_value
    if peeled.locked_metric:
        intent.locked_metric = peeled.locked_metric
        intent.primary_variable = peeled.locked_metric
    if peeled.locked_group:
        intent.locked_group = peeled.locked_group
        intent.grouping_variable = peeled.locked_group
    if peeled.force_intent == "CORRELATIONS" and peeled.locked_metric and peeled.locked_group:
        # The relationship analysis reads predictor_variables, not metric/group.
        intent.predictor_variables = [peeled.locked_metric, peeled.locked_group]
    from clinical_analytics.core.nl_query_config import TIER_1_PATTERN_MATCH_THRESHOLD

    # The peel's intent only replaces a parse too weak to stand; confidence is untouched.
    if peeled.force_intent and intent.confidence < TIER_1_PATTERN_MATCH_THRESHOLD:
        intent.intent_type = peeled.force_intent
    return intent


def unapplied_constraints(
    query: str,
    plan: Any,
    columns: list[str] | None = None,
) -> list[str]:
    """Constraints the question stated that the plan does not carry.

    Gender words are matched as inflections of a stored value. Every other
    check is a token that appears in exactly one column name, or a cutoff
    operator. There is no list of clinical events.
    """
    text = " ".join(query.lower().split())
    gaps: list[str] = []
    steps = list(getattr(plan, "cohort_steps", []) or []) if plan is not None else []
    used = _plan_columns(plan, steps)
    comparing = re.search(r"\bcompare\b|\bby sex\b|\bby gender\b", text) is not None

    over = _OVER.search(text)
    # "not over 65" is applied as "65 or under"; a negation that may not reach it is never applied.
    over_scope = _scope(text, over.start()) if over else "clear"
    over_op = "le" if over_scope == "negated" else "gt"
    if over and (
        over_scope == "uncertain"
        or not any(
            step.op == over_op and str(step.value) in {over.group(1), str(float(over.group(1)))} for step in steps
        )
    ):
        gaps.append(f"over {over.group(1)}")
    if _DECILE.search(text) and not any(step.op == "ge_percent_rank" for step in steps):
        gaps.append("decile")
    if _MEDIAN_CUT.search(text) and not any(step.op == "gt_median" for step in steps):
        gaps.append("median cutoff")

    tokens = _tokens(text)
    if columns and not comparing:
        # Sex is applied by grouping on it or by counting a share/rate of it, not only by a step.
        grouped = any(
            _is_group(str(getattr(plan, attr, "") or "")) for attr in ("group_by", "proportion_column", "rate_column")
        )
        for word, inflections in (("women", {"woman", "women"}), ("men", {"man", "men"})):
            if (
                tokens & inflections
                and not grouped
                and not any(step.op in {"eq", "ne", "in"} and _is_group(step.column) for step in steps)
            ):
                if any(_is_group(column) for column in columns):
                    gaps.append(word)
                    break
        owners = _token_owners(columns)
        for token, owners_for_token in owners.items():
            if token not in tokens or len(owners_for_token) != 1:
                continue
            column = owners_for_token[0]
            if column not in used and token not in _STOP and token not in NO_LIMIT_WORDS:
                gaps.append(token)
    if (
        re.search(r"\bhow many\b", text)
        and not steps
        and not getattr(plan, "rate_column", None)
        and not getattr(plan, "proportion_column", None)
    ):
        used_tokens: set[str] = set()
        for column in used:
            used_tokens |= _tokens(str(column))
        # Words word accounting treats as carrying no limit ("are there", "patients", "taking")
        # are not gaps: the two checks share one vocabulary. Name the words themselves: the
        # user sees this list.
        grouping = {"by", "each", "per"} if getattr(plan, "group_by", None) else set()
        gaps.extend(sorted(tokens - used_tokens - NO_LIMIT_WORDS - grouping))
    return gaps


def _plan_columns(plan: Any, steps: list[CohortStep]) -> set[str]:
    used = {step.column for step in steps}
    if plan is None:
        return used
    for attr in ("metric", "group_by", "proportion_column", "rate_column"):
        current = getattr(plan, attr, None)
        if current:
            used.add(str(current))
    return used


def _token_owners(columns: list[str]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for column in columns:
        for token in _tokens(column):
            if len(token) < 3 and token not in {"ldl", "hdl", "cd4", "icu", "age", "sex", "bmi", "dod"}:
                continue
            owners.setdefault(token, []).append(column)
    return owners


def _value_hits(
    text: str,
    columns: list[str],
    values_for: Callable[[str], list[Any]],
) -> list[tuple[str, Any, int]]:
    """(column, value, position) of each stored value the question names.

    The value is a list when the question joins several values of one column with "or"
    ("in the north or west"): any of them.
    """
    hits: list[tuple[str, Any, int]] = []
    tokens = _tokens(text, digits=True)
    for column in columns:
        # Rank (exact word, inflection, prefix), then specificity (more words of the value
        # named: "North East" over "North"). Two values tied on both is ambiguous: bind
        # neither, so the unbound word holds the plan. Never decide by value order.
        found_values: list[tuple[int, int, int, Any, str]] = []
        stored = values_for(column)
        candidates: list[tuple[str, Any]] = [
            (str(value).strip().lower(), value) for value in stored if value is not None and not isinstance(value, bool)
        ]
        # Coded columns name their values in the header ("DEXA result 1: Normal 2: Osteopenia").
        for code, code_label in (parse_column_name(column).value_mapping or {}).items():
            code_value = _code_value(stored, code)
            if code_value is not None:
                candidates.append((code_label.strip().lower(), code_value))
        for label, value in candidates:
            # A stored number is never named by a word: "2.2" splits into the token "2", which
            # would bind "type 2 diabetes" to Creatinine = 2.2. Numbers bind only through an
            # explicit comparison.
            if not label or _NUMBER.fullmatch(label):
                continue
            found = _label_at(text, tokens, label)
            if found is None:
                continue
            found_values.append((found[0], -len(_tokens(label, digits=True)), found[1], value, label))
        if not found_values:
            continue
        found_values.sort(key=lambda item: (item[0], item[1]))
        best = found_values[0]
        tied = [item for item in found_values if item[:2] == best[:2]]
        if len(tied) > 1:
            joined = _joined_by_or(text, tied)
            if joined is not None:
                hits.append((column, joined, min(item[2] for item in tied)))
            continue
        hits.append((column, best[3], best[2]))
    hits.sort(key=lambda item: item[2])
    return hits


def _joined_by_or(text: str, tied: list[tuple[int, int, int, Any, str]]) -> list[Any] | None:
    """The tied values when the question lists them with "or" ("north or west"), else None.

    Only whole-word values (rank 0 or 1), each at its own place, with nothing but "or" and
    words that carry no limit between neighbours. "north and south" is not a list of
    alternatives: it can mean both together or each one, so it stays unbound.
    """
    if tied[0][0] > 1 or len({item[2] for item in tied}) != len(tied):
        return None
    ordered = sorted(tied, key=lambda item: item[2])
    for left, right in zip(ordered, ordered[1:], strict=False):
        gap = text[left[2] : right[2]]
        words = re.findall(r"[a-z0-9]+", gap)[len(re.findall(r"[a-z0-9]+", left[4])) :]
        if "or" not in words or any(word not in NO_LIMIT_WORDS and word != "or" for word in words):
            return None
    return [item[3] for item in ordered]


def _label_at(text: str, tokens: set[str], label: str) -> tuple[int, int] | None:
    """(rank, position) of a stored value named in the question, or None.

    Rank 0 is every word of the value, 1 an English inflection, 2 a word that starts a
    word of the value ("diabet" for "diabetes") or shares its stem ("northern" for "North").
    A word inside another word does not count: "male" is inside "female" and must never
    select it. A value with a one-letter or number word ("drug A", "Type 2") is named only
    by those words in that order: "on a drug" does not name "drug A".
    """
    from clinical_analytics.core.word_accounting import stem_match

    words = re.findall(r"[a-z0-9]+", label)
    if len(words) > 1 and any(len(word) == 1 or word.isdigit() for word in words):
        # "type 2" is not inside "type 2.2": the label ends where its last word ends.
        phrase = re.search(r"\b" + r"[\s_-]+".join(re.escape(word) for word in words) + r"\b(?![.,]\d)", text)
        return (0, phrase.start()) if phrase else None
    label_tokens = _tokens(label, digits=True)
    if label_tokens and label_tokens <= tokens:
        return 0, min(_word_at(text, word) for word in label_tokens)
    inflections = _VALUE_INFLECTIONS.get(label, set())
    shared = inflections & tokens
    if shared:
        return 1, min(_word_at(text, word) for word in shared)
    prefixed = [
        token
        for token in tokens
        if len(token) >= 4
        and not token.isdigit()
        and any(part.startswith(token) or stem_match(token, part) for part in label_tokens)
    ]
    if prefixed:
        return 2, min(_word_at(text, token) for token in prefixed)
    return None


def _word_at(text: str, word: str) -> int:
    match = re.search(rf"\b{re.escape(word)}\b", text)
    return match.start() if match else text.find(word)


def _binary_polarity(
    column: str,
    values: list[Any],
    text: str,
    at: int,
) -> tuple[str, Any] | None:
    """The limit naming a yes/no column sets: ("eq", yes), or ("eq", no) when negated.

    A text column that records absence as a value ("statin": atorvastatin, simvastatin,
    none) is named the same way: "on statins" is ("ne", "none"), "not on a statin" is
    ("eq", "none").
    """
    yes, no = yes_no_values(column, values)
    scope = _scope(text, at)
    if scope == "uncertain":
        return None
    negated = scope == "negated"
    if yes is not None and no is not None:
        return ("eq", no) if negated else ("eq", yes)
    absent = absence_value(values)
    if absent is None or not _names_a_subset(text, at):
        return None
    return ("eq", absent) if negated else ("ne", absent)


_ASKING = frozenset({"which", "what", "whose"})


def _names_a_subset(text: str, at: int) -> bool:
    """The column word picks out patients ("on statins", "diabetic patients", "without
    diabetes"), rather than asking about its values ("which statin") or naming a variable
    ("bmi, statin use")."""
    before = re.findall(r"[a-z0-9]+", text[:at])
    word = re.match(r"[a-z0-9]+", text[at:])
    after = re.findall(r"[a-z0-9]+", text[at + (word.end() if word else 0) :])
    if before and (before[-1] in _ASKING):
        return False
    if after and after[0] in SUBJECT_WORDS:
        return True
    return bool(before) and (before[-1] in NO_LIMIT_WORDS or before[-1] in NEGATION_CUES)


_ABSENCE_LABELS = frozenset({"none", "no", "never"})


def absence_value(values: list[Any]) -> Any:
    """The stored value meaning "none of these" on a text column with two or more others."""
    texts = [value for value in values if isinstance(value, str)]
    absent = [value for value in texts if value.strip().lower() in _ABSENCE_LABELS]
    if len(absent) != 1 or len(texts) - 1 < 2 or len(texts) != len([v for v in values if v is not None]):
        return None
    return absent[0]


# Words that may stand right before a yes/no column without changing what it means.
_PLAIN_LEAD = frozenset(
    {"how", "many", "number", "count", "total", "proportion", "share", "percentage", "percent", "fraction"}
    | set(STATISTIC_WORDS)
)


def _qualified(text: str, at: int, columns: list[str], values_for: Callable[[str], list[Any]]) -> bool:
    """True when an unknown word qualifies the column word ("exclude unknown status").

    "status" alone could be read as status = yes; "unknown status" is something else, so the
    column is not bound as yes/no and the qualifying word holds the plan. A word that names a
    column or a stored value ("female smokers") is a limit of its own, not a qualifier.
    """
    before = re.findall(r"[a-z0-9]+", text[:at])
    if not before:
        return False
    word = before[-1]
    if word in NO_LIMIT_WORDS or word in _PLAIN_LEAD or word.isdigit() or word in NEGATION_CUES:
        return False
    if any(_tokens(word) & _tokens(column) for column in columns):
        return False
    return not _value_hits(word, columns, values_for)


def _scope(text: str, at: int) -> NegationScope:
    """Negation scope of the word at ``at``, by the rule the gate and the justifier share
    (question_vocabulary.negation_scope); "statin non-users" negates "statin" from after it."""
    from clinical_analytics.core.word_accounting import tokenize

    before = negation_scope(tokenize(text[:at]))
    if not postfix_negated(tokenize(text[at:])):
        return before
    # "not statin non-users": two negations on one word; the words do not settle it.
    return "negated" if before == "clear" else "uncertain"


def yes_no_values(column: str, values: list[Any]) -> tuple[Any, Any]:
    present = [value for value in values if value is not None]
    for value in present:
        if value is True:
            yes = value
            no = next((item for item in present if item is False), None)
            return yes, no
    labels = {str(value).strip().lower(): value for value in present}
    if "yes" in labels and "no" in labels:
        return labels["yes"], labels["no"]
    parsed = parse_column_name(column)
    mapping = parsed.value_mapping or {}
    if mapping:
        # A header codebook says which code is which ("0: Yes 1: No" is not an indicator).
        yes_code = next((code for code, label in mapping.items() if label.strip().lower() == "yes"), None)
        no_code = next((code for code, label in mapping.items() if label.strip().lower() == "no"), None)
        if yes_code is None or no_code is None:
            return None, None
        return _code_value(present, yes_code), _code_value(present, no_code)
    return _indicator_values(present) or (None, None)


def _indicator_values(present: list[Any]) -> tuple[Any, Any] | None:
    """(1, 0) for a 0/1 indicator column ("diabetes", "outcome", "event"), else None.

    Only whole numbers 0 and 1, both stored, and nothing else: a column holding 0, 1 and 2,
    or only 1, is not read as yes/no. A column with a header codebook never reaches here.
    """
    numbers = [value for value in present if isinstance(value, int | float) and not isinstance(value, bool)]
    if len(numbers) != len(present) or {float(value) for value in numbers} != {0.0, 1.0}:
        return None
    yes = next(value for value in numbers if float(value) == 1.0)
    no = next(value for value in numbers if float(value) == 0.0)
    return yes, no


def _code_value(values: list[Any], code: str) -> Any:
    for value in values:
        if str(value).strip() == code:
            return value
        try:
            if float(value) == float(code):
                return value
        except (TypeError, ValueError):
            continue
    return None


def _temporal_value(values: list[Any]) -> bool | None:
    samples = [value for value in values if value is not None]
    if not samples:
        return None
    dated = 0
    for value in samples[:8]:
        if re.search(r"\d{4}-\d{2}-\d{2}", str(value)):
            dated += 1
    if dated >= max(1, len(samples[:8]) // 2):
        return True
    return None


def _mentioned(text: str, columns: list[str], tokens: set[str]) -> list[tuple[str, int]]:
    """Columns the question names, with where. A word names a column word when the gate would
    say so too (word_accounting.stem_match): "diabetic" names "diabetes", "events" "event"."""
    from clinical_analytics.core.word_accounting import stem_match

    found: list[tuple[str, int, int, set[str]]] = []
    for column in columns:
        column_tokens = _tokens(column)
        # Exact words first (the scoring the parser has always used); a stem only when no
        # word of the column is written out ("diabetic" for "diabetes").
        named = column_tokens & tokens
        words = set(named)
        if not named:
            words = {word for word in tokens if any(stem_match(word, part) for part in column_tokens)}
            named = {part for part in column_tokens if any(stem_match(word, part) for word in words)}
        if not words:
            continue
        score = len(named) * 3 - len(column_tokens - named)
        if score <= 0:
            continue
        # Where the column is named: its own word, not a linking word it shares ("use").
        anchors = [word for word in words if word not in NO_LIMIT_WORDS] or list(words)
        at = min(_word_at(text, word) for word in anchors)
        found.append((column, score, at, words))
    found.sort(key=lambda item: -item[1])
    claimed: set[str] = set()
    kept: list[tuple[str, int]] = []
    for column, _score_value, at, words in found:
        if words & claimed:
            continue
        claimed |= words
        kept.append((column, at))
    return kept


def _ranked(columns: list[str], tokens: set[str]) -> list[tuple[str, int]]:
    scored = [(column, _score(column, tokens)) for column in columns]
    scored.sort(key=lambda item: -item[1])
    return scored


def _best_column(columns: list[str], tokens: set[str]) -> str | None:
    if not tokens:
        return None
    ranked = _ranked(columns, tokens)
    if not ranked or ranked[0][1] <= 0:
        return None
    return ranked[0][0]


def _score(column: str, tokens: set[str]) -> int:
    alias = _tokens(column)
    if not alias:
        return 0
    matched = alias & tokens
    if not matched:
        return 0
    extra = alias - tokens
    return len(matched) * 3 - len(extra)


def _tokens(text: str, digits: bool = False) -> set[str]:
    """Content words. ``digits=True`` keeps numbers, so stored values "Type 1" and "Type 2" differ."""
    parts = re.findall(r"[a-z0-9]+", text.lower())
    words = {
        part
        for part in parts
        if part not in _STOP and (digits or not part.isdigit()) and (len(part) > 1 or (digits and part.isdigit()))
    }
    # Plurals name the same column or value ("statins", "smokers", "females").
    words |= {word[:-1] for word in words if len(word) > 4 and word.endswith("s") and not word.endswith("ss")}
    return words - _STOP


def _comparisons(lowered: str) -> list[tuple[re.Match[str], str]]:
    """Every "<direction> N" / "N <direction>" in the question, with its operator."""
    found: list[tuple[re.Match[str], str]] = []
    for match in _PRE_COMPARE.finditer(lowered):
        direction = match.group("dir").lower()
        op = _PRE_OPS.get(direction) or ("lt" if direction.startswith(_LOWER_WORDS) else "gt")
        found.append((match, op))
    taken = {match.start("num") for match, _op in found}
    for match in _POST_COMPARE.finditer(lowered):
        if match.start("num") in taken:
            continue
        direction = match.group("dir").lower()
        found.append((match, "le" if any(word in direction for word in _LOWER_WORDS) else "ge"))
    found.sort(key=lambda item: item[0].start())
    return found


def _number_owner(
    lowered: str,
    match: re.Match[str],
    columns: list[str],
    values_for: Callable[[str], list[Any]],
) -> str | None:
    """Column the compared number belongs to (shared rule with the word-accounting gate)."""
    tokens = _owner_tokens(lowered[: match.end("num")])
    # The word after the number can be its unit ("over 100 kg").
    following = _owner_tokens(lowered[match.end("num") :])[:2]
    return _owner_of(tokens, columns, values_for, following)


def _owner_tokens(text_to_number: str) -> list[str]:
    from clinical_analytics.core.word_accounting import tokenize

    return tokenize(text_to_number)


def _owner_of(
    tokens: list[str],
    columns: list[str],
    values_for: Callable[[str], list[Any]],
    following: list[str] | None = None,
) -> str | None:
    """Owner of the number that ends ``tokens``, by the word-accounting gate's own rule."""
    from clinical_analytics.core.word_accounting import _is_measure, number_owner, numeric_ranges

    numeric = {column for column in columns if _is_measure(values_for(column))}
    return number_owner(
        [*tokens, *(following or [])], len(tokens) - 1, columns, numeric, ranges=numeric_ranges(columns, values_for)
    )


def _is_group(column: str) -> bool:
    return bool(_tokens(column) & {"sex", "gender"})


def _bound_tokens(steps: list[CohortStep], columns: list[str]) -> set[str]:
    bound: set[str] = set()
    for step in steps:
        bound |= _tokens(step.column)
    return bound


def _drop_tokens(text: str, drop: set[str]) -> str:
    def keep(match: re.Match[str]) -> str:
        word = match.group(0)
        return " " if word.lower() in drop else word

    return re.sub(r"[A-Za-z0-9]+", keep, text)


def _cleanup(text: str) -> str:
    text = re.sub(r"\b(for|among)\s+patients\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bon\s+a\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ?.,")
    text = re.sub(r"^(among|for|of)\b[\s,]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(for|among|of)\b\s*$", "", text, flags=re.IGNORECASE)
    text = text.strip(" ?.,")
    text = re.sub(r"\s+", " ", text).strip(" ,")
    if not text:
        return "how many patients?"
    if not text.endswith("?"):
        text += "?"
    return text
