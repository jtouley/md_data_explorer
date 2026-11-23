# SPDX-License-Identifier: MIT
"""A small unit grammar (UCUM-style unit atoms) for column names and dictionary text (ADR017).

A name ends in a unit when its last tokens are unit atoms and a stem remains before them:
``grip_kg`` is kilograms, ``urea_mmol_l`` is mmol/L, ``dose_mg_kg_d`` is mg/kg/day. Atoms
after the first divide it, as written units do. A single atom stands alone only when it cannot
be read as anything else: one letter (``vitamin_d``), a singular time word (``visit_day``, a day
number), ``hr`` (a heart rate), ``min`` (a minimum) or a token that reads as a timepoint
(``m2`` = month 2) is a unit only inside a ratio (``kg_m2``). Nothing
else is guessed: a dictionary's stated unit always wins (``contracts.dictionary``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

# Unit atoms (lower case, as name tokens are) and how they are written.
_MASS: Final = {"kg": "kg", "g": "g", "mg": "mg", "mcg": "mcg", "ug": "µg", "ng": "ng", "pg": "pg"}
_AMOUNT: Final = {
    "mol": "mol", "mmol": "mmol", "umol": "µmol", "nmol": "nmol", "pmol": "pmol", "meq": "mEq", "mosm": "mOsm",
    "iu": "IU", "u": "U", "kcal": "kcal",
}  # fmt: skip
_VOLUME: Final = {"l": "L", "dl": "dL", "ml": "mL", "ul": "µL", "mm3": "mm³"}
_LENGTH: Final = {"km": "km", "m": "m", "cm": "cm", "mm": "mm", "m2": "m²", "cm2": "cm²", "mm2": "mm²"}
_PRESSURE: Final = {"mmhg": "mmHg", "cmh2o": "cmH2O", "kpa": "kPa"}
_OTHER: Final = {"pct": "%", "percent": "%", "bpm": "beats/min", "lb": "lb", "lbs": "lb", "cells": "cells"}
# Time atoms: (as a unit of its own, as the divisor of a ratio).
_TIME: Final = {
    "s": ("seconds", "s"), "sec": ("seconds", "s"), "secs": ("seconds", "s"), "seconds": ("seconds", "s"),
    "min": ("minutes", "min"), "mins": ("minutes", "min"), "minutes": ("minutes", "min"),
    "h": ("hours", "h"), "hr": ("hours", "h"), "hrs": ("hours", "h"), "hours": ("hours", "h"),
    "d": ("days", "day"), "day": ("days", "day"), "days": ("days", "day"),
    "wk": ("weeks", "week"), "wks": ("weeks", "week"), "week": ("weeks", "week"), "weeks": ("weeks", "week"),
    "mo": ("months", "month"), "month": ("months", "month"), "months": ("months", "month"),
    "yr": ("years", "year"), "yrs": ("years", "year"), "year": ("years", "year"), "years": ("years", "year"),
}  # fmt: skip
ATOMS: Final[dict[str, str]] = {**_MASS, **_AMOUNT, **_VOLUME, **_LENGTH, **_PRESSURE, **_OTHER}
# Atoms that are never a unit on their own (see the module docstring); every other atom can be.
_NOT_ALONE: Final = frozenset({"g", "l", "m", "u", "s", "cells", "min", "h", "hr", "d", "day", "wk", "week",
                               "mo", "month", "yr", "year"})  # fmt: skip
_PER: Final = frozenset({"per"})
# A token that also reads as a timepoint ("m2" = month 2, "w4", "d7", "y1"): a timepoint at the
# end of a name, never a unit on its own (an area unit then needs a ratio or the dictionary).
_TIMEPOINT_LIKE: Final = re.compile(r"^(?:m|mo|w|wk|d|y|yr|t|v)\d{1,3}$")


def is_atom(token: str) -> bool:
    """A unit atom ("mg", "dl", "min", "cmh2o")."""
    return token.lower() in ATOMS or token.lower() in _TIME


def _written(token: str, *, divisor: bool) -> str:
    if token in _TIME:
        alone, per = _TIME[token]
        return per if divisor else alone
    return ATOMS[token]


def unit_from_tokens(tokens: Sequence[str]) -> str | None:
    """The unit the last tokens of a name spell, or None (no unit, or nothing before it)."""
    tail: list[str] = []
    for token in reversed(tokens):
        if is_atom(token) or (token in _PER and tail):
            tail.insert(0, token)
            continue
        break
    while tail and tail[0] in _PER:
        tail.pop(0)
    atoms = [t for t in tail if t not in _PER]
    if not atoms or len(tail) >= len(tokens):
        return None  # no stem before the unit ("kg" alone, "mg_dl")
    if len(atoms) == 1:
        alone = atoms[0] not in _NOT_ALONE and not _TIMEPOINT_LIKE.match(atoms[0])
        return _written(atoms[0], divisor=False) if alone else None
    first, *rest = atoms
    return "/".join([_written(first, divisor=False), *(_written(t, divisor=True) for t in rest)])
