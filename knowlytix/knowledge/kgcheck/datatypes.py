# SPDX-License-Identifier: Apache-2.0
"""Literal datatypes of the audit schema: parsing a stored token as a typed value.

A relation's ``value.datatype`` names how its object tokens are read. The
lexical forms are those of the XSD types of the same name, so native checks
and SHACL validation (CON007) accept the same tokens: a decimal has no
exponent, a boolean is ``true``, ``false``, ``1`` or ``0``, and a date may carry
a time zone. Parsing keeps the token exactly as stored and reports why it is
not a value of the type, so a rule can quote the lexical form. Nothing is
coerced: ``"5,000"`` is not a decimal and ``"1.0"`` is not an integer. Numbers
compare as decimals, never as floats, so bounds hold exactly at their edges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from knowlytix.knowledge.kgcheck.models import DeclKey, SemanticsDecl

DATATYPES = ("decimal", "integer", "date", "boolean", "string")
_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")  # xsd:decimal: no exponent
_INTEGER = re.compile(r"[+-]?\d+")
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:Z|[+-](?:0\d|1[0-3]):[0-5]\d|[+-]14:00)?")
_NONFINITE = re.compile(r"[+-]?(?:nan|inf|infinity)", re.I)


VALUE_KEYS = (DeclKey.DATATYPE, DeclKey.MINIMUM, DeclKey.MAXIMUM, DeclKey.ALLOWED_VALUE)


def value_relations(sem: SemanticsDecl) -> tuple[str, ...]:
    """Relations with a literal value contract (datatype, bounds or allowed values), sorted."""
    return tuple(sorted({d.subject for k in VALUE_KEYS for d in sem.of(k)}))


def effective_datatype(sem: SemanticsDecl, relation: str) -> str:
    """The datatype a relation's objects are read as: declared, else decimal when only
    bounds are declared, else string. CON002 and the SHACL translation both use it."""
    stated = sem.find(DeclKey.DATATYPE, relation)
    if stated is not None:
        return stated.value
    bounded = any(sem.find(k, relation) for k in (DeclKey.MINIMUM, DeclKey.MAXIMUM))
    return "decimal" if bounded else "string"


@dataclass(frozen=True)
class TypedValue:
    """A token read under a datatype: the value, or the reason it is not one."""

    lexical: str
    datatype: str
    value: Decimal | date | bool | str | None = None
    problem: str | None = None  # "nonfinite" or "lexical"


def parse_typed(lexical: str, datatype: str) -> TypedValue:
    """``lexical`` read as ``datatype`` (one of ``DATATYPES``)."""
    if datatype == "string":
        return TypedValue(lexical, datatype, lexical)
    if datatype in ("decimal", "integer"):
        return _number(lexical, datatype)
    if datatype == "boolean":
        known = {"true": True, "false": False, "1": True, "0": False}.get(lexical)
        return TypedValue(lexical, datatype, known, None if known is not None else "lexical")
    if datatype == "date":
        return _date(lexical)
    raise ValueError(f"unknown datatype {datatype!r}")


def _number(lexical: str, datatype: str) -> TypedValue:
    if _NONFINITE.fullmatch(lexical):
        return TypedValue(lexical, datatype, problem="nonfinite")
    pattern = _DECIMAL if datatype == "decimal" else _INTEGER
    if not pattern.fullmatch(lexical):
        return TypedValue(lexical, datatype, problem="lexical")
    return TypedValue(lexical, datatype, Decimal(lexical))


def _date(lexical: str) -> TypedValue:
    if (m := _DATE.fullmatch(lexical)):
        try:
            return TypedValue(lexical, "date", date.fromisoformat(m[1]))
        except ValueError:
            pass
    return TypedValue(lexical, "date", problem="lexical")


def is_number(token: str) -> bool:
    """True when ``token`` is a finite decimal numeral as written (no grouping commas)."""
    return bool(_DECIMAL.fullmatch(token))


def same_value(a: str, b: str) -> bool:
    """Value equality of two tokens: numerically when both are decimals, else exactly."""
    if _DECIMAL.fullmatch(a) and _DECIMAL.fullmatch(b):
        return Decimal(a) == Decimal(b)
    return a == b


def value_key(token: str) -> str:
    """A key equal for tokens that ``same_value`` treats as equal."""
    if _DECIMAL.fullmatch(token):
        number = Decimal(token)
        return "n:" + ("0" if number == 0 else format(number.normalize(), "f"))
    return "s:" + token
