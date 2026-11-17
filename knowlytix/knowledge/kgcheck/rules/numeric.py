# SPDX-License-Identifier: Apache-2.0
"""Numeric rules NUM001-NUM003: ENM slots against numeric facts; phase ranges; units,
scales, periods and qualifiers of cited quantities.

OntoDisc writes one exact-numeric-memory (ENM) slot per (relation, head) whose
triples state a number. Several different values for one slot cannot share it:
stores written since 47782ab omit the slot and record the values in
``metadata.json`` ``numeric_conflicts``; older stores silently kept the last
value written. The persisted ENM holds values only, not the integrity hashes
of the in-memory graph, so these checks establish consistency between
artifacts, not the historical authenticity of a value.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from knowlytix.knowledge.kgcheck.models import (
    DeclKey,
    Finding,
    GraphView,
    NumericConflict,
    NumericSlot,
    RawRef,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import is_numeric_token
from knowlytix.knowledge.kgcheck.quantities import DOLLARS, compatible_unit, quantities
from knowlytix.knowledge.kgcheck.registry import (
    Outcome,
    RuleContext,
    RuleSpec,
    artifact_prerequisite,
    builtin,
)
from knowlytix.knowledge.kgcheck.rules._common import rows_by_assertion, triple

Key = tuple[str, str]  # (relation, head), as the store keys ENM
_VIEWS = (GraphView.STORED, GraphView.CONFORMANT)


@dataclass
class _Slot:
    values: dict[Decimal, list[str]] = field(default_factory=dict)  # value -> assertion ids
    enm: NumericSlot | None = None
    conflict: NumericConflict | None = None
    surface: list[str] = field(default_factory=list)  # non-canonical tails, e.g. "$5,000"


def _slots(ctx: RuleContext) -> dict[Key, _Slot]:
    slots: dict[Key, _Slot] = {}
    surfaces: dict[Key, list[str]] = {}
    for aid in rows_by_assertion(ctx.snapshot, ctx.view):
        h, r, o = triple(ctx.snapshot, aid)
        if is_numeric_token(o):
            slots.setdefault((r, h), _Slot()).values.setdefault(Decimal(o), []).append(aid)
        elif any(c.isdigit() for c in o):
            surfaces.setdefault((r, h), []).append(o)
    for slot in ctx.snapshot.numeric_slots:
        slots.setdefault((slot.relation, slot.head), _Slot()).enm = slot
    for conflict in ctx.snapshot.numeric_conflicts:
        slot = slots.setdefault((conflict.relation, conflict.head), _Slot())
        slot.conflict, slot.surface = conflict, surfaces.get((conflict.relation, conflict.head), [])
    return slots


# --- NUM001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "NUM001", "7", "numeric", "Numeric memory against numeric facts", Severity.ERROR,
    prerequisites=(artifact_prerequisite("enm.json"),), views=_VIEWS,
    parameters={"relative_tolerance": 1e-9}, parameter_bounds={"relative_tolerance": (0.0, 1.0)},
    basis="numeric triples of the view grouped by (relation, head) and compared with enm.json "
          "slots and metadata.json numeric_conflicts; each decimal source value is compared "
          "exactly with the number written in enm.json, and a difference within "
          "relative_tolerance is precision loss, beyond it a mismatch"))
def num001(ctx: RuleContext) -> Outcome:
    slots = _slots(ctx)
    tolerance = float(ctx.params["relative_tolerance"])
    findings = [f for key, slot in sorted(slots.items()) for f in _check(ctx, key, slot, tolerance)]
    if (ctx.snapshot.metadata.get("numeric_memory_policy") in ("exclude_document_identifiers_v1", "typed_quantities_v1")
            and not isinstance(ctx.snapshot.metadata.get("numeric_memory_excluded_relations"), Mapping)):
        findings.append(ctx.finding("declared numeric-memory policy has no applied relation list",
                                    discriminator="policy_relations_missing",
                                    locations=(RawRef("metadata.json", "/numeric_memory_excluded_relations"),),
                                    suggestion="persist the relation-to-kind list applied by the producer"))
    return Outcome(eligible=len(slots), findings=tuple(findings), metrics={
        "numeric_keys": len(slots), "enm_slots": len(ctx.snapshot.numeric_slots),
        "recorded_conflicts": len(ctx.snapshot.numeric_conflicts),
        "conflicts_recorded_by_producer": ctx.snapshot.capabilities.numeric_conflicts_recorded})


def _check(ctx: RuleContext, key: Key, slot: _Slot, tolerance: float) -> Iterator[Finding]:
    n = len(slot.values)
    exclusions = ctx.snapshot.metadata.get("numeric_memory_exclusions", ())
    policy = ctx.snapshot.metadata.get("numeric_memory_excluded_relations", {})
    excluded = (ctx.snapshot.metadata.get("numeric_memory_policy") in ("exclude_document_identifiers_v1", "typed_quantities_v1")
                and isinstance(policy, Mapping) and key[0] in policy)
    if ctx.snapshot.metadata.get("numeric_memory_policy") == "typed_quantities_v1":
        declarations = ctx.snapshot.metadata.get("quantity_relations", {})
        spec = declarations.get(key[0]) if isinstance(declarations, Mapping) else None
        valid = (isinstance(spec, Mapping) and spec.get("kind") in
                 ("amount", "duration", "count", "ratio", "measurement")
                 and isinstance(spec.get("unit"), str) and bool(spec["unit"].strip())
                 and spec.get("scope") == "head" and spec.get("unconditional") is True)
        if not valid and not excluded:
            yield _finding(ctx, key, slot, "untyped_quantity", "has no valid declared quantity type, unit and scope")
    recorded = set()
    if excluded:
        for e in exclusions if isinstance(exclusions, tuple) else ():
            if (isinstance(e, Mapping) and e.get("relation") == key[0] and e.get("head") == key[1]
                    and e.get("kind") == policy[key[0]]):
                try:
                    value = Decimal(str(e.get("value")))
                    if value.is_finite():
                        recorded.add(value)
                except InvalidOperation:
                    pass
        if slot.enm is not None:
            yield _finding(ctx, key, slot, "metadata_slot", "has an ENM slot despite its "
                           "declared numeric admission exclusion")
        if _unrecorded(set(slot.values), recorded):
            yield _finding(ctx, key, slot, "metadata_unrecorded", "has excluded values without "
                           "matching (relation, head, value) exclusion records", Severity.WARNING)
        if slot.conflict is not None and slot.surface:
            yield _finding(ctx, key, slot, "recorded_conflict", f"is recorded as a conflict over "
                           f"{n + len(slot.surface)} stated values, some written in surface form "
                           f"({', '.join(sorted(slot.surface)[:5])}) that kgcheck does not parse",
                           Severity.INFO)
            return
        if n >= 2 and slot.conflict is None:
            yield _finding(ctx, key, slot, "excluded_multi_value", f"states {n} different values; "
                           "its relation is excluded from numeric memory, so no conflict is "
                           "recorded and a reader can answer any one of them")
            return
        if not _unrecorded(set(slot.values), recorded) and n and slot.enm is None:
            yield _finding(ctx, key, slot, "metadata_excluded", "is retained in the graph "
                           "and intentionally excluded from quantity memory", Severity.INFO)
    if slot.enm is not None and slot.conflict is not None:
        yield _finding(ctx, key, slot, "slot_and_conflict", "has an ENM slot although "
                       "metadata.json records it as a conflict")
    if n >= 2:
        yield from _several(ctx, key, slot)
    elif n == 1:
        if not (excluded and slot.enm is None and slot.conflict is None):
            yield from _single(ctx, key, slot, tolerance)
    elif slot.enm is not None:
        severity = Severity.WARNING if ctx.view is GraphView.STORED else Severity.INFO
        yield _finding(ctx, key, slot, "orphan_slot", f"has an ENM slot ({slot.enm.lexical}) but "
                       f"no numeric triple in the {ctx.view.value} view", severity)
    elif slot.conflict is not None:
        severity = Severity.WARNING if ctx.view is GraphView.STORED else Severity.INFO
        yield _finding(ctx, key, slot, "orphan_conflict", "is recorded as a conflict but has no "
                       f"numeric triple in the {ctx.view.value} view", severity)


def _unrecorded(values: set[Decimal], recorded: set[Decimal]) -> set[Decimal]:
    """Values with no exclusion record. The producer records a value as ``str(float)``,
    so a record matches when it is equal exactly or as a double."""
    doubles = {float(r) for r in recorded}
    return {v for v in values if v not in recorded and float(v) not in doubles}


def _several(ctx: RuleContext, key: Key, slot: _Slot) -> Iterator[Finding]:
    count = len(slot.values)
    if slot.enm is not None:
        yield _finding(ctx, key, slot, "silent_overwrite", f"has {count} candidate values but ENM "
                       f"keeps one, {slot.enm.lexical}; the others are lost")
    elif slot.conflict is not None:
        agrees = set(slot.conflict.values) == {float(v) for v in slot.values}
        yield _finding(ctx, key, slot, "recorded_conflict",
                       f"has {count} candidate values; the slot is omitted and the conflict recorded"
                       + ("" if agrees else ", but the recorded values differ from the view's"),
                       Severity.INFO if agrees else Severity.WARNING)
    else:
        yield _finding(ctx, key, slot, "unrecorded_conflict", f"has {count} candidate values, no "
                       "ENM slot and no recorded conflict", Severity.WARNING)


def _single(ctx: RuleContext, key: Key, slot: _Slot, tolerance: float) -> Iterator[Finding]:
    (value,) = slot.values
    if slot.conflict is not None:
        yield _finding(ctx, key, slot, "conflict_single_value", "is recorded as a conflict but "
                       f"the {ctx.view.value} view states one value", Severity.WARNING)
    elif slot.enm is None:
        severity = Severity.WARNING if ctx.view is GraphView.STORED else Severity.INFO
        yield _finding(ctx, key, slot, "missing_slot", f"states {value} but has no ENM slot", severity)
    elif value != (stored := stored_decimal(slot.enm)):
        yield from _compare(ctx, key, slot, value, stored, tolerance)


def stored_decimal(slot: NumericSlot) -> Decimal:
    """The ENM value exactly as written in enm.json, or the double's shortest repr."""
    try:
        return Decimal(slot.lexical)
    except InvalidOperation:
        return Decimal(repr(slot.value))


def _compare(ctx: RuleContext, key: Key, slot: _Slot, value: Decimal, stored: Decimal,
             tolerance: float) -> Iterator[Finding]:
    difference = abs(value - stored)
    if difference <= Decimal(repr(tolerance)) * max(Decimal(1), abs(value)):
        yield _finding(ctx, key, slot, "precision_loss", f"states {value}; ENM stores {stored}, "
                       f"{difference} apart, within the relative tolerance", Severity.INFO)
    else:
        yield _finding(ctx, key, slot, "value_mismatch", f"states {value} but ENM stores {stored}")


def _finding(ctx: RuleContext, key: Key, slot: _Slot, code: str, message: str,
             severity: Severity | None = None) -> Finding:
    relation, head = key
    refs = [r for r in (slot.enm.raw if slot.enm else None,
                        slot.conflict.raw if slot.conflict else None) if r is not None]
    return ctx.finding(f"numeric slot {relation}::{head} {message}", severity=severity,
                       affected=tuple(sorted({a for ids in slot.values.values() for a in ids})),
                       locations=tuple(refs) or (RawRef("enm.json"),), witness=_witness(slot),
                       discriminator=f"{code}:{relation}::{head}", suggestion=_SUGGESTION[code])


def _witness(slot: _Slot) -> dict[str, object]:
    return {"view_values": [str(v) for v in sorted(slot.values)],
            "enm_value": slot.enm.lexical if slot.enm else None,
            "recorded_conflict": [repr(v) for v in slot.conflict.values] if slot.conflict else None}


_SUGGESTION = {
    "untyped_quantity": "declare an explicit quantity kind, unit and unconditional head scope, or exclude the relation from scalar memory",
    "metadata_unrecorded": "record each excluded identifier value; do not reuse a record for another version",
    "metadata_slot": "remove the identifier from quantity memory or correct the declared policy",
    "metadata_excluded": "use the graph's typed metadata rather than a quantitative lookup",
    "excluded_multi_value": "give each value distinguishing context (tier, date, condition); "
                            "rebuilding does not help while the relation is excluded",
    "orphan_conflict": "the recorded conflict has no supporting facts in this view; regenerate "
                       "metadata.json numeric_conflicts",
    "slot_and_conflict": "regenerate numeric memory; a slot cannot be both kept and ambiguous",
    "silent_overwrite": "rebuild with a current OntoDisc, which omits ambiguous slots and records "
                        "them, or give the facts distinguishing context",
    "recorded_conflict": "add context (tier, date, condition) so each value has its own key",
    "unrecorded_conflict": "rebuild numeric memory; the omission is not recorded",
    "conflict_single_value": "regenerate metadata.json numeric_conflicts",
    "missing_slot": "exact-value lookups for this fact fall back to the geometry",
    "orphan_slot": "the slot has no supporting fact in this view; review its origin",
    "precision_loss": "no action needed unless exact decimal recall matters",
    "value_mismatch": "numeric memory disagrees with the graph; rebuild it",
}


# --- NUM002 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "NUM002", "2", "numeric", "Phase encoder ranges", Severity.ERROR,
    prerequisites=(artifact_prerequisite("phase.json"),), views=_VIEWS,
    basis="each phase.json range must have v_min < v_max; numeric facts of a relation named like "
          "an encoder (with or without a has_ prefix) are compared with its range, and the "
          "number compared is reported"))
def num002(ctx: RuleContext) -> Outcome:
    values: dict[str, list[tuple[Decimal, str]]] = {}
    for aid in rows_by_assertion(ctx.snapshot, ctx.view):
        h, r, o = triple(ctx.snapshot, aid)
        if is_numeric_token(o):
            for name in {r, r.removeprefix("has_")}:
                values.setdefault(name, []).append((Decimal(o), aid))
    findings: list[Finding] = []
    for rng in ctx.snapshot.phase_ranges:
        if rng.v_min >= rng.v_max:
            findings.append(ctx.finding(f"phase encoder {rng.name!r} has v_min {rng.v_min} >= v_max "
                                        f"{rng.v_max}", locations=(rng.raw,), discriminator=rng.name,
                                        suggestion="an encoder needs a non-empty range"))
            continue
        outside = [(v, a) for v, a in values.get(rng.name, []) if not rng.v_min <= v <= rng.v_max]
        if outside:
            findings.append(ctx.finding(
                f"{len(outside)} {rng.name} values lie outside [{rng.v_min}, {rng.v_max}] and "
                "would be clipped", severity=Severity.WARNING, affected=tuple(a for _, a in outside),
                locations=(rng.raw,), witness={"values": [str(v) for v, _ in outside[:20]]},
                discriminator=f"clipped:{rng.name}"))
    compared = sum(len(values.get(rng.name, ())) for rng in ctx.snapshot.phase_ranges)
    return Outcome(eligible=len(ctx.snapshot.phase_ranges), findings=tuple(findings),
                   metrics={"values_compared": compared})


# --- NUM003 -------------------------------------------------------------------------
_LOCALIZED = frozenset({"sentence", "line"})


@dataclass
class _Observed:
    """What the citations of one numeric assertion say around its value."""

    units: set[str] = field(default_factory=set)
    periods: set[str] = field(default_factory=set)
    qualifiers: set[str] = field(default_factory=set)
    scaled: bool = False  # the value matches a numeral with its magnitude word applied
    unscaled: list[str] = field(default_factory=list)  # matches only without the magnitude
    links: list[RawRef] = field(default_factory=list)


@builtin(RuleSpec(
    "NUM003", "2", "numeric", "Units, scales, periods and qualifiers", Severity.WARNING, phase="C",
    prerequisites=("provenance",), views=_VIEWS,
    basis="numerals equal to each numeric object in its sentence and line citations, read with "
          "the currency or unit, magnitude word, period and bound words written next to them "
          "(kgcheck.quantities); compared with the relation's declared unit, period and "
          "qualifier (audit schema value block, else metadata.json quantity_relations units) "
          "and across the relation's facts. A bare number has no unit; nothing is inferred. "
          "Facts whose citations do not state the value are counted, not checked"))
def num003(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    declared = _declared_units(ctx)
    links: dict[str, list] = {}
    for link in snap.evidence:
        links.setdefault(link.assertion_id, []).append(link)
    findings, by_relation, counts = [], {}, {"no_citation": 0, "section_level_only": 0,
                                              "value_not_found": 0, "bare": 0}
    eligible = 0
    for aid in rows_by_assertion(snap, ctx.view):
        h, r, o = triple(snap, aid)
        if not is_numeric_token(o):
            continue
        cited = [lk for lk in links.get(aid, ()) if lk.kind in _LOCALIZED]
        if not cited:
            counts["section_level_only" if links.get(aid) else "no_citation"] += 1
            continue
        seen = _observe(Decimal(o), cited)
        if not seen.links:  # nothing to read the unit from; EVD002 reports the citation
            counts["value_not_found"] += 1
            continue
        eligible += 1
        counts["bare"] += not seen.units
        by_relation.setdefault(r, []).append((aid, seen))
        findings += _assertion_findings(ctx, aid, (h, r, o), seen, declared.get(r, {}))
    findings += [f for r, seen in sorted(by_relation.items())
                 for f in _relation_findings(ctx, r, seen, declared.get(r, {}))]
    return Outcome(eligible=eligible, findings=tuple(findings),
                   metrics={**counts, "localized": eligible,
                            "declared_relations": sorted(declared)})


def _declared_units(ctx: RuleContext) -> dict[str, dict[str, str]]:
    """relation token -> declared unit/period/qualifier, with where the unit came from."""
    sem, out = ctx.snapshot.semantics.accepted(), {}
    quantity = ctx.snapshot.metadata.get("quantity_relations")
    for relation, spec in (quantity.items() if isinstance(quantity, Mapping) else ()):
        if isinstance(spec, Mapping) and isinstance(spec.get("unit"), str) and spec["unit"].strip():
            out[relation] = {"unit": spec["unit"].strip(), "unit_source": "metadata.json"}
    for key, name in ((DeclKey.UNIT, "unit"), (DeclKey.PERIOD, "period"),
                      (DeclKey.QUALIFIER, "qualifier")):
        for d in sem.of(key):
            token = d.subject.partition(":")[2]
            out.setdefault(token, {})[name] = d.value
            if key is DeclKey.UNIT:
                out[token]["unit_source"] = "audit_schema.json"
    return out


def _observe(value: Decimal, cited: list) -> _Observed:
    seen = _Observed()
    for link in cited:
        text = link.sentence if link.kind == "sentence" else link.span
        for q in quantities(text):
            if q.scaled == value:
                seen.scaled = True
            elif q.value == value and q.scaled != q.value:
                seen.unscaled.append(q.text)
            else:
                continue
            seen.links.append(link.raw)
            seen.units |= {q.unit} if q.unit else set()
            seen.periods |= {q.period} if q.period else set()
            seen.qualifiers |= {q.qualifier} if q.qualifier else set()
    return seen


def _assertion_findings(ctx: RuleContext, aid: str, t: tuple[str, str, str], seen: _Observed,
                        declared: Mapping[str, str]) -> Iterator[Finding]:
    witness = {"triple": list(t), "units": sorted(seen.units), "periods": sorted(seen.periods),
               "qualifiers": sorted(seen.qualifiers), "declared": dict(declared)}
    if seen.unscaled and not seen.scaled:
        yield _quantity(ctx, aid, seen, "scale", f"{list(t)} keeps {t[2]} although its citation "
                        f"states {seen.unscaled[0]!r}", witness, "store the scaled value")
    unit = declared.get("unit")
    if unit and seen.units and not any(compatible_unit(unit, u) for u in seen.units):
        yield _quantity(ctx, aid, seen, "unit", f"{list(t)}: the citation states "
                        f"{', '.join(sorted(seen.units))}, the relation declares {unit}", witness,
                        "correct the fact or the declared unit")
    elif len({_currency_family(u) for u in seen.units}) > 1:
        yield _quantity(ctx, aid, seen, "units_disagree", f"{list(t)}: its citations state "
                        f"different units ({', '.join(sorted(seen.units))})", witness,
                        "check which citation states this fact")
    for key, values in (("period", seen.periods), ("qualifier", seen.qualifiers)):
        expected = declared.get(key)
        if expected and values and expected not in values:
            yield _quantity(ctx, aid, seen, key, f"{list(t)}: the citation states "
                            f"{key} {', '.join(sorted(values))}, the relation declares {expected}",
                            witness, f"check whether the fact carries the declared {key}")


def _currency_family(unit: str) -> str:
    return "dollar" if unit in DOLLARS else unit


def _relation_findings(ctx: RuleContext, relation: str, seen: list[tuple[str, _Observed]],
                       declared: Mapping[str, str]) -> Iterator[Finding]:
    units = sorted({u for _, s in seen for u in s.units})
    if declared.get("unit") or not units:
        return
    families = {_currency_family(u) for u in units}
    rid = f"relation:{relation}"
    if len(families) > 1:
        yield ctx.finding(f"facts of {relation!r} cite values in different units: "
                          f"{', '.join(units)}", affected=(rid,), discriminator="mixed_units",
                          witness={"relation": relation, "units": units,
                                   "facts": [a for a, _ in seen][:50]},
                          suggestion="split the relation by unit, or declare one and correct the "
                                     "facts that differ")
    else:
        yield ctx.finding(f"facts of {relation!r} cite values in {units[0]} but the relation "
                          "declares no unit", severity=Severity.INFO, affected=(rid,),
                          discriminator="undeclared_unit", witness={"relation": relation,
                                                                    "units": units},
                          suggestion="declare value.unit in the audit schema")


def _quantity(ctx: RuleContext, aid: str, seen: _Observed, code: str, message: str,
              witness: dict, suggestion: str) -> Finding:
    return ctx.finding(message, affected=(aid,), locations=tuple(seen.links[:5]),
                       discriminator=code, witness=witness, suggestion=suggestion)
