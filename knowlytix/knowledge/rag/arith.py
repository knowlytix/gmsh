# SPDX-License-Identifier: Apache-2.0
"""Arithmetic over cited values for the ``facts`` route (``RagConfig.facts_arithmetic``).

A question like "how many more days does a bank get when no suspect is identified?"
has no fact that states the answer: it is the difference of two values the facts
give (60 days and 30 days). Left to the answering model, such a subtraction is
unreliable, and a model told to answer only "exactly as asked" declines. Here the
model only chooses which quantities to combine; code checks that their units agree,
that the operation is the one asked and that a difference isn't negative, computes
the result exactly, and the answer cites every fact a value came from.

The model's choice of quantities is not checked beyond that: a pair with the right
units but the wrong meaning gives a wrong number, cited. Keep the setting off where
that risk isn't acceptable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from knowlytix.knowledge.rag.retrieve import RetrievedFact

__all__ = ["ARITH_SYSTEM", "Op", "Quantity", "Result", "compute", "detect", "listing",
           "asked_unit", "needs_question_value", "parse_expression", "quantities", "sentence",
           "worth_asking"]

Op = Literal["difference", "sum"]

# A question asks for arithmetic only in these forms (see :func:`detect`): "how
# many/much [<unit>] more/fewer/less/longer/shorter/later/earlier ...", "how many/much
# ... remain/remaining/left", "how many/much ... in total/total/combined/altogether/in
# all", and "difference between", "sum of", "total of" when the question also has a
# "how many/much" or a value. Bare "more than", "left" or "remaining" are not enough
# ("the remaining balance threshold" asks for a value the document states), and
# "how many <things>" counts things, unless the things are a unit ("how many days").
_HOW = r"\bhow (?:many|much)\b"
_COMPARE = r"(?:more|fewer|less|longer|shorter|later|earlier)"
_UNIT_WORDS = (r"(?:calendar days?|business days?|days?|weeks?|months?|years?|hours?"
               r"|percent|dollars?)")
_DIFFERENCE = re.compile(
    _HOW + r"(?: " + _UNIT_WORDS + r")? " + _COMPARE + r"\b"
    r"|" + _HOW + r"[^?]*\b(?:remains?|remaining|left)\b|\bdifference between\b",
    re.IGNORECASE)
_REMAIN = re.compile(_HOW + r"[^?]*\b(?:remains?|remaining|left)\b", re.IGNORECASE)
# Not bare "total"/"in total": "how many days in total does a bank have" asks for a
# value the document states (60), not a sum of every day value (90).
_SUM = re.compile(_HOW + r"[^?]*\b(?:combined|altogether)\b"
                  r"|\b(?:sum|total) of\b", re.IGNORECASE)
# What "how many" asks to count: a unit (days), a comparative (more ...), or a thing.
# ("total", "additional", "extra" are skipped too: "how many total days" asks for days.
# They don't ask for arithmetic themselves: "how many additional days may a bank get"
# asks for a value the document states.)
_COUNTED = re.compile(r"\bhow many (?:(?:" + _COMPARE + r"|total|additional|extra) )?(\w+)(?: (\w+))?",
                      re.IGNORECASE)
_NOT_A_NOUN = {"more", "fewer", "less", "longer", "shorter", "later", "earlier", "of",
               "total", "additional", "extra",
               "are", "is", "do", "does", "did", "can", "will", "would", "should"}

_UNITS = {   # surface -> unit (calendar days are days; business days are not)
    "calendar days": "days", "calendar day": "days", "days": "days", "day": "days",
    "business days": "business days", "business day": "business days",
    "weeks": "weeks", "week": "weeks", "months": "months", "month": "months",
    "years": "years", "year": "years", "hours": "hours", "hour": "hours",
    "%": "percent", "percent": "percent",
}
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"   # 5,000 or 5000; not a trailing comma
_UNIT_RE = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
# A number with a unit after it, or a dollar amount. Bare numbers (section numbers,
# years, counts of things) are not quantities here.
_QUANTITY = re.compile(
    rf"(?P<dollar>\$\s?)(?P<dv>{_NUMBER})"
    rf"|(?P<n>{_NUMBER})[\s-]*(?P<unit>{_UNIT_RE})(?![a-z])", re.IGNORECASE)

ARITH_SYSTEM = (
    "You are given numbered QUANTITIES, each taken from a fact, its source sentence or "
    "the question, and a question that asks for a difference or a total. Reply with "
    "ONLY an expression over the quantity labels that answers the question, like "
    "'Q1 - Q2' or 'Q1 + Q3' (two quantities for a difference, two or more for a "
    "total). Reply NONE if the quantities don't answer it, or if one quantity "
    "already answers it on its own. No other words."
)


@dataclass(frozen=True)
class Quantity:
    """A value with its unit and where it came from: the fact at ``fact`` (its tail,
    or its source sentence when ``in_source``), or the question (``fact`` is None).
    ``text`` is the value as written."""

    value: Decimal
    unit: str
    text: str
    fact: int | None
    in_source: bool = False


@dataclass(frozen=True)
class Result:
    op: Op
    operands: tuple[int, ...]   # quantity indices, in expression order
    value: Decimal
    unit: str


def detect(question: str) -> Op | None:
    """Whether the question asks for a difference or a total of values. Not when
    "how many" counts things ("how many SARs ... in total"), and not for "difference
    between" / "sum of" / "total of" without a "how many/much" or a value in the
    question ("is there a difference between a SAR and a CTR?")."""
    counted = _COUNTED.search(question)
    if counted:
        word = counted.group(1).lower()
        pair = f"{word} {(counted.group(2) or '').lower()}".strip()
        unit = any(re.fullmatch(_UNIT_WORDS, w, re.IGNORECASE) for w in (word, pair))
        if word not in _NOT_A_NOUN and not unit:
            return None                        # counts things, not values
    numeric = bool(re.search(_HOW, question, re.IGNORECASE)) or bool(_scan(question))
    if _DIFFERENCE.search(question) and numeric:
        return "difference"
    if _SUM.search(question) and numeric:
        return "sum"
    return None


def asked_unit(question: str) -> str | None:
    """The unit "how many <unit>" asks for ("how many calendar days" -> days), or None
    when the question doesn't name one ("how much longer")."""
    counted = _COUNTED.search(question)
    if not counted:
        return None
    word = counted.group(1).lower()
    pair = f"{word} {(counted.group(2) or '').lower()}".strip()
    for surface in (pair, word):
        if surface in _UNITS:
            return _UNITS[surface]
        if surface in ("dollar", "dollars"):
            return "dollars"
    return None


def needs_question_value(question: str) -> bool:
    """A "how many ... remain/left" question subtracts from a value the question
    gives ("...files 90 days after... how many days remain?"); without one, two of the
    document's values would answer a question that didn't ask about them."""
    return bool(_REMAIN.search(question))


def _scan(text: str) -> list[tuple[Decimal, str, str]]:
    out = []
    for m in _QUANTITY.finditer(text or ""):
        if m.group("dollar") is not None:
            out.append((Decimal(m.group("dv").replace(",", "")), "dollars", m.group(0).strip()))
        else:
            out.append((Decimal(m.group("n").replace(",", "")),
                        _UNITS[m.group("unit").lower()], m.group(0).strip()))
    return out


def quantities(question: str, facts: list[RetrievedFact]) -> list[Quantity]:
    """Every quantity in the facts' tails, then their source sentences, then the
    question. A value a source sentence gives is credited to a fact whose tail states
    it when there is one; each (value, unit, fact) once."""
    found: list[Quantity] = []
    seen: set[tuple[Decimal, str, int | None]] = set()

    def add(q: Quantity) -> None:
        if (q.value, q.unit, q.fact) not in seen:
            seen.add((q.value, q.unit, q.fact))
            found.append(q)
    for i, f in enumerate(facts):
        for value, unit, text in _scan(f.tail):
            add(Quantity(value, unit, text, i))
    stated = {(q.value, q.unit) for q in found}
    for i, f in enumerate(facts):
        for value, unit, text in _scan(f.raw or ""):
            if (value, unit) not in stated:
                add(Quantity(value, unit, text, i, in_source=True))
    for value, unit, text in _scan(question):
        add(Quantity(value, unit, text, None))
    return found


def worth_asking(qs: list[Quantity]) -> bool:
    """Whether the model could combine anything: two quantities share a unit, and at
    least one of them comes from the document."""
    by_unit: dict[str, list[Quantity]] = {}
    for q in qs:
        by_unit.setdefault(q.unit, []).append(q)
    return any(len(g) >= 2 and any(q.fact is not None for q in g) for g in by_unit.values())


def listing(qs: list[Quantity]) -> str:
    """The quantities as the model sees them, labelled Q1, Q2, ..."""
    def where(q: Quantity) -> str:
        if q.fact is None:
            return "from the question"
        return f"from {'the source of ' if q.in_source else ''}fact [{q.fact + 1}]"
    return "\n".join(f"Q{n} = {q.text} ({where(q)})" for n, q in enumerate(qs, start=1))


_EXPR = re.compile(r"^\s*Q(\d+)(?:\s*([-+])\s*Q(\d+))+\s*$", re.IGNORECASE)
_TERM = re.compile(r"([-+])?\s*Q(\d+)", re.IGNORECASE)
_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})


def parse_expression(reply: str, n: int) -> tuple[Op, tuple[int, ...]] | None:
    """``'Q1 - Q3'`` -> ("difference", (0, 2)); ``'Q1 + Q2 + Q4'`` -> a sum. Only one
    subtraction of two quantities, or a sum of two or more, each used once; None for
    anything else (including NONE). A Unicode minus or dash counts as a minus."""
    text = (reply or "").translate(_MINUS).strip()
    line = text.splitlines()[0].strip().strip("`'\".") if text else ""
    if not _EXPR.match(line):
        return None
    terms = [(sign or "+", int(k) - 1) for sign, k in _TERM.findall(line)]
    idx = tuple(k for _, k in terms)
    if any(not 0 <= k < n for k in idx) or len(set(idx)) != len(idx):
        return None
    signs = [s for s, _ in terms[1:]]
    if signs == ["-"]:
        return "difference", idx
    if signs and all(s == "+" for s in signs):
        return "sum", idx
    return None


def compute(op: Op, operands: tuple[int, ...], qs: list[Quantity], asked: Op) -> Result | None:
    """The result, exactly (decimal), when the operation is the one the question
    asks for, every operand has the same unit and a difference isn't negative;
    else None."""
    if op != asked:
        return None
    units = {qs[k].unit for k in operands}
    if len(units) != 1:
        return None
    values = [qs[k].value for k in operands]
    value = values[0] - values[1] if op == "difference" else sum(values, Decimal(0))
    if value < 0:
        return None   # "how many more/remain" asks for the larger minus the smaller
    return Result(op=op, operands=operands, value=value, unit=units.pop())


def _amount(value: Decimal, unit: str) -> str:
    """A value with its unit, as a person writes it: $5,000, 15%, 30 days."""
    exponent = value.as_tuple().exponent
    places = max(0, -exponent) if isinstance(exponent, int) else 0
    number = f"{value:,.{places}f}"
    if unit == "dollars":
        return f"${number}"
    if unit == "percent":
        return f"{number}%"
    singular = unit[:-1] if value == 1 and unit.endswith("s") else unit
    return f"{number} {singular}"


def sentence(result: Result, qs: list[Quantity], cite: dict[int, int]) -> str:
    """The answer, written by code: the result, then how it was computed, each value
    with the citation of its fact (``cite`` maps a fact to its number among the
    answer's sources) or marked as given in the question."""
    def term(k: int) -> str:
        q = qs[k]
        shown = _amount(q.value, q.unit)
        return f"{shown} [{cite[q.fact]}]" if q.fact is not None else f"{shown} (from the question)"
    joiner = " minus " if result.op == "difference" else " plus "
    return f"{_amount(result.value, result.unit)}: {joiner.join(term(k) for k in result.operands)}."
