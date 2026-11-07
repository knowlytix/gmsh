# SPDX-License-Identifier: Apache-2.0
"""Artifact rules ART001-ART005: contract, internal consistency, views, duplicates, stability."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cache

from knowlytix.knowledge.kgcheck.audit_schema import SIDECAR
from knowlytix.knowledge.kgcheck.contract import artifact_contract, artifact_specs
from knowlytix.knowledge.kgcheck.frozen import FrozenJson, thaw
from knowlytix.knowledge.kgcheck.gate import SNAPSHOT_CATEGORY
from knowlytix.knowledge.kgcheck.literals import nonquantity_numbers
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    Finding,
    GraphView,
    RawRef,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import entity_resolution_key, triple_key
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import (
    artifact,
    mapping,
    quarantine_reasons,
    rows_by_assertion,
    sequence,
    triple,
    triple_views,
)
from knowlytix.knowledge.kgcheck.sources import split_sections
from knowlytix.knowledge.kgcheck.validate import Validator

SNAPSHOT_CODES = frozenset({"changed_during_read", "disappeared_during_read"})
_SENTENCE_END = re.compile(r"[.;:!?]\s*$")


# --- ART001 -----------------------------------------------------------------------
@cache
def _validator() -> Validator:
    return Validator(artifact_contract())


def _contract_problems(snapshot: AuditSnapshot) -> Iterator[tuple[RawRef, str, str]]:
    for spec in artifact_specs():
        if spec.filename in snapshot.parsed:
            for v in _validator().errors(snapshot.parsed[spec.filename],
                                         {"$ref": f"#/$defs/{spec.definition}"}):
                yield RawRef(spec.filename, v.pointer), "contract_violation", v.message


def _blank_tokens(snapshot: AuditSnapshot) -> Iterator[tuple[RawRef, str, str]]:
    for view in (GraphView.STORED, GraphView.CONFORMANT):
        for row in snapshot.view(view):
            for k, token in enumerate(triple(snapshot, row.assertion_id)):
                if not token.strip():
                    yield (RawRef(row.raw.artifact, f"{row.raw.pointer}/{k}"), "blank_token",
                           "a triple token is empty or whitespace")


@builtin(RuleSpec(
    "ART001", "5", "artifact", "Artifact contract and parse problems", Severity.ERROR,
    basis="reader diagnostics (except the audit schema sidecar's, see META001), validation "
          "against the ontodisc-artifacts contract, and blank-token checks on stored and "
          "conformant rows"))
def art001(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    problems: dict[RawRef, list[tuple[str, str]]] = {}
    reader = ((d.location, d.code, d.message) for d in snap.diagnostics
              if d.code not in SNAPSHOT_CODES and d.location.artifact != SIDECAR)  # META001's
    for ref, code, message in [*reader, *_contract_problems(snap), *_blank_tokens(snap)]:
        problems.setdefault(ref, []).append((code, message))
    findings = [ctx.finding("; ".join(m for _, m in items), locations=(ref,),
                            witness={"codes": sorted({c for c, _ in items})},
                            suggestion="repair or regenerate the artifact; the audit read it as is")
                for ref, items in problems.items()]
    return Outcome(eligible=len(snap.artifacts), findings=tuple(findings),
                   metrics={"artifacts_parsed": len(snap.parsed),
                            "locations_with_problems": len(problems)})


# --- ART002 -----------------------------------------------------------------------
@dataclass(frozen=True)
class _Issue:
    message: str
    ref: RawRef
    discriminator: str
    affected: tuple[str, ...] = ()
    severity: Severity = Severity.ERROR
    witness: Mapping[str, object] | None = None


def _index_map(adapter: Mapping[str, FrozenJson], namespace: str) -> Iterator[_Issue]:
    to_idx, from_idx = mapping(adapter.get(f"{namespace}_to_idx")), mapping(adapter.get(f"idx_to_{namespace}"))
    name = "adapter.json"
    for token, idx in to_idx.items():
        if from_idx.get(str(idx)) != token:
            yield _Issue(f"{namespace}_to_idx[{token!r}] = {idx}, but idx_to_{namespace}[{idx}] is "
                         f"{from_idx.get(str(idx))!r}", RawRef(name, f"/{namespace}_to_idx"),
                         f"inverse:{namespace}:{token}", (f"{_ns(namespace)}:{token}",))
    for idx, token in from_idx.items():
        if to_idx.get(token) != _int(idx):
            yield _Issue(f"idx_to_{namespace}[{idx}] = {token!r} has no matching {namespace}_to_idx entry",
                         RawRef(name, f"/idx_to_{namespace}"), f"inverse_back:{namespace}:{idx}")
    yield from _index_range(adapter, to_idx, namespace)


def _index_range(adapter: Mapping[str, FrozenJson], to_idx: Mapping[str, FrozenJson],
                 namespace: str) -> Iterator[_Issue]:
    indices = [i for i in to_idx.values() if isinstance(i, int)]
    count_key = _COUNT_KEY[namespace]
    count = adapter.get(count_key)
    if sorted(indices) != list(range(len(indices))):
        repeated = sorted(i for i, n in Counter(indices).items() if n > 1)
        yield _Issue(f"{namespace} indices are not 0..{len(indices) - 1} each used once",
                     RawRef("adapter.json", f"/{namespace}_to_idx"), f"range:{namespace}",
                     witness={"repeated": repeated, "max": max(indices, default=None)})
    if count != len(to_idx):
        yield _Issue(f"{count_key} is {count!r} but {namespace}_to_idx has {len(to_idx)} entries",
                     RawRef("adapter.json", f"/{count_key}"), f"count:{namespace}")


_COUNT_KEY = {"entity": "num_entities", "relation": "num_relations"}


def _ns(namespace: str) -> str:
    return "relation" if namespace == "relation" else "entity"


def _int(text: str) -> int | None:
    return int(text) if text.isdigit() else None


def _model_dims(snapshot: AuditSnapshot, adapter: Mapping[str, FrozenJson]) -> Iterator[_Issue]:
    dims = artifact(snapshot, "model_dims.json")
    for key in ("num_entities", "num_relations"):
        if key in dims and key in adapter and dims[key] != adapter[key]:
            yield _Issue(f"model_dims.json {key} is {dims[key]} but adapter.json says {adapter[key]}",
                         RawRef("model_dims.json", f"/{key}"), f"dims:{key}")


def _vocabulary(snapshot: AuditSnapshot, adapter: Mapping[str, FrozenJson]) -> Iterator[_Issue]:
    if not snapshot.has_view(GraphView.STORED):
        return
    used: dict[str, set[str]] = {"entity": set(), "relation": set()}
    for aid in rows_by_assertion(snapshot, GraphView.STORED):
        h, r, o = triple(snapshot, aid)
        used["entity"] |= {h, o}
        used["relation"].add(r)
    for namespace in ("entity", "relation"):
        vocab = set(mapping(adapter.get(f"{namespace}_to_idx")))
        yield from _membership(namespace, used[namespace], vocab)


def _membership(namespace: str, used: set[str], vocab: set[str]) -> Iterator[_Issue]:
    ref = RawRef("adapter.json", f"/{namespace}_to_idx")
    for token in sorted(used - vocab):
        yield _Issue(f"stored {namespace} {token!r} has no index in the trained model's vocabulary",
                     ref, f"untrained:{namespace}:{token}", (f"{namespace}:{token}",))
    for token in sorted(vocab - used):
        yield _Issue(f"trained {namespace} {token!r} occurs in no stored triple", ref,
                     f"unused:{namespace}:{token}", (f"{namespace}:{token}",), Severity.INFO)


def _closed_schema(snapshot: AuditSnapshot) -> Iterator[_Issue]:
    closed = artifact(snapshot, "closed_schema.json")
    if not snapshot.has_view(GraphView.CONFORMANT):
        return
    rows = [triple(snapshot, aid) for aid in rows_by_assertion(snapshot, GraphView.CONFORMANT)]
    derived = {"entities": {t for h, _, o in rows for t in (h, o)}, "relations": {r for _, r, _ in rows}}
    for key, tokens in derived.items():
        listed = {t for t in sequence(closed.get(key)) if isinstance(t, str)}
        if listed != tokens:
            yield _Issue(f"closed_schema.json {key} differ from the tokens of its triples",
                         RawRef("closed_schema.json", f"/{key}"), f"closed:{key}",
                         witness={"listed_only": sorted(listed - tokens)[:50],
                                  "triples_only": sorted(tokens - listed)[:50]})
    yield from _closed_counts(closed, {"conformant": len(snapshot.view(GraphView.CONFORMANT)),
                                       "entities": len(sequence(closed.get("entities"))),
                                       "relations": len(sequence(closed.get("relations")))})


def _closed_counts(closed: Mapping[str, FrozenJson], actual: dict[str, int]) -> Iterator[_Issue]:
    counts = mapping(closed.get("counts"))
    for key, n in actual.items():
        if key in counts and counts[key] != n:
            yield _Issue(f"closed_schema.json counts.{key} is {counts[key]} but {n} were found",
                         RawRef("closed_schema.json", f"/counts/{key}"), f"counts:{key}")


@builtin(RuleSpec(
    "ART002", "6", "artifact", "Consistency of indices, vocabularies and counts", Severity.ERROR,
    prerequisites=("artifact:adapter.json",),
    basis="adapter index maps and their inverses, model_dims, stored-view vocabulary against the "
          "trained vocabulary, closed_schema lists and counts recomputed from its triples, and "
          "rules.json records against their assertions, spans, clauses and retained source; "
          "kgcheck's own recognizers, not the producer's code"))
def art002(ctx: RuleContext) -> Outcome:
    snap, adapter = ctx.snapshot, artifact(ctx.snapshot, "adapter.json")
    issues = [*_index_map(adapter, "entity"), *_index_map(adapter, "relation"),
              *_model_dims(snap, adapter), *_vocabulary(snap, adapter), *_closed_schema(snap),
              *_coverage_metadata(snap), *_rule_records(snap)]
    return Outcome(eligible=_checks(snap), findings=tuple(_issue_finding(ctx, i) for i in issues))


def _names(value: FrozenJson) -> list[str] | FrozenJson:
    """A recorded name list sorted for comparison; anything else as recorded (ART001 flags it)."""
    if isinstance(value, tuple) and all(isinstance(v, str) for v in value):
        return sorted(value)
    return thaw(value)


def _coverage_metadata(snapshot):
    recorded = snapshot.metadata.get("extraction_coverage")
    if not isinstance(recorded, Mapping) or not snapshot.sources or not snapshot.has_view(GraphView.STORED):
        return
    sections = split_sections(snapshot.sources[0].text)
    selected = set(rows_by_assertion(snapshot, GraphView.STORED))
    cited = {e.section for e in snapshot.evidence if e.assertion_id in selected and e.section in sections}
    expected = {"sections": len(sections), "sections_with_admitted_facts": len(cited),
                "uncited_sections": sorted(set(sections) - cited)}
    actual = {key: _names(recorded.get(key)) if key == "uncited_sections" else recorded.get(key)
              for key in expected}
    if actual != expected:
        yield _Issue("recorded extraction coverage differs from retained source and admitted evidence",
                     RawRef("metadata.json", "/extraction_coverage"), "extraction_coverage",
                     witness={"recorded": actual, "recomputed": expected})


def _rule_records(snapshot):
    """Validate links and source offsets independently of producer parsing."""
    if "rules.json" not in snapshot.parsed:
        return
    delivered = {triple(snapshot, aid) for aid in rows_by_assertion(snapshot, GraphView.CONFORMANT)}
    seen = set()
    for i, raw in enumerate(sequence(artifact(snapshot, "rules.json").get("rules"))):
        record = mapping(raw)
        ref = RawRef("rules.json", f"/rules/{i}")
        row = sequence(record.get("triple"))
        text = record.get("text")
        identity = record.get("id")
        if (len(row) != 3 or any(not isinstance(v, str) for v in row)
                or tuple(row) not in delivered or text != row[2]
                or record.get("subject") != row[0]):
            yield _Issue("rule does not refer to its retained conformant assertion", ref, f"rule_assertion:{i}")
        if not isinstance(identity, str) or identity in seen:
            yield _Issue("rule identity is missing or duplicated", ref, f"rule_identity:{i}")
        if isinstance(identity, str):
            seen.add(identity)
        if not isinstance(text, str):
            continue
        spans = [record.get(k) for k in ("actor", "modality", "action") if record.get(k) is not None]
        spans += list(sequence(record.get("conditions"))) + list(sequence(record.get("quantities")))
        for raw_link in sequence(record.get("relationships")):
            link = mapping(raw_link)
            spans.extend([link.get("from"), link.get("to")])
        if "nonquantity_numbers" in record:
            if thaw(record.get("nonquantity_numbers")) != nonquantity_numbers(text):
                yield _Issue("numeric context annotations do not match the source literal", ref, f"numeric_context:{i}")
        spans += list(sequence(record.get("nonquantity_numbers")))
        quantities = sequence(record.get("quantities"))
        clause_ids = set()
        attached_quantities = []
        clause_records = sequence(record.get("clauses"))
        for j, raw_clause in enumerate(clause_records):
            clause = mapping(raw_clause)
            if (clause.get("scope_status") == "parsed" and (clause.get("review_reasons")
                    or clause.get("action") is None
                    or (clause.get("modality") is None and clause.get("kind") != "descriptive")
                    or (clause.get("actor") is None and not clause.get("context_actor")))):
                yield _Issue("parsed clause still has unresolved fields", ref, f"clause_readiness:{i}:{j}")
            if clause.get("context_actor") is not None and not _context_actor_supported(
                    snapshot, clause, j, row, sequence(record.get("evidence"))):
                yield _Issue("context actor is not supported by its exact source sentence", ref,
                             f"context_actor:{i}:{j}")
            cid = clause.get("id")
            if not isinstance(cid, str) or cid in clause_ids:
                yield _Issue("clause identity is missing or duplicated", ref, f"clause_identity:{i}:{j}")
            if isinstance(cid, str):
                clause_ids.add(cid)
            parent = mapping(clause.get("span"))
            children = [clause.get(k) for k in ("actor", "modality", "action") if clause.get(k) is not None]
            children += list(sequence(clause.get("qualifiers")))
            spans.extend([parent, *children])
            for child_index, child in enumerate(children):
                child = mapping(child)
                if (type(parent.get("start")) is int and type(parent.get("end")) is int
                        and type(child.get("start")) is int and type(child.get("end")) is int
                        and not parent["start"] <= child["start"] < child["end"] <= parent["end"]):
                    yield _Issue("clause field lies outside its clause", ref, f"clause_scope:{i}:{j}:{child_index}")
            for attachment, index in enumerate(sequence(clause.get("quantity_indices"))):
                attached_quantities.append(index)
                quantity = mapping(quantities[index]) if type(index) is int and 0 <= index < len(quantities) else {}
                if (not quantity or type(parent.get("start")) is not int or type(parent.get("end")) is not int
                        or type(quantity.get("start")) is not int or type(quantity.get("end")) is not int
                        or not parent["start"] <= quantity["start"] < quantity["end"] <= parent["end"]):
                    yield _Issue("quantity attachment lies outside its clause", ref, f"clause_quantity:{i}:{j}:{attachment}")
        if "clauses" in record:
            indices = [n for n in attached_quantities if type(n) is int]
            if sorted(indices) != list(range(len(quantities))) or len(indices) != len(attached_quantities):
                yield _Issue("clause quantity attachments are missing or duplicated", ref, f"clause_quantities:{i}")
            pending = any(mapping(c).get("scope_status") != "parsed" for c in clause_records)
            expected = "requires_review" if pending or not clause_records else "single_clause" if len(clause_records) == 1 else "multiple_clauses"
            if record.get("scope_status") != expected:
                yield _Issue("rule readiness differs from its child clauses", ref, f"rule_readiness:{i}")
        for j, raw_span in enumerate(spans):
            span = mapping(raw_span)
            start, end = span.get("start"), span.get("end")
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or text[start:end] != span.get("text")):
                yield _Issue("rule span does not match retained assertion text", ref, f"rule_span:{i}:{j}")
        for j, raw_quantity in enumerate(sequence(record.get("quantities"))):
            quantity = mapping(raw_quantity)
            if quantity.get("scope") != identity or quantity.get("scalar_memory_eligible") is not False:
                yield _Issue("quantity has lost its parent rule scope", ref, f"rule_quantity_scope:{i}:{j}")


def _context_actor_supported(snapshot: AuditSnapshot, clause: Mapping[str, FrozenJson], j: int,
                             row: tuple[FrozenJson, ...], citations: tuple[FrozenJson, ...]) -> bool:
    """The first clause has no actor of its own; its context actor is the exact text at its
    offsets in a cited sentence retained in the source, starts that sentence, names the rule's
    head, and is followed by the rule text (ART006 also compares it with the producer's own
    resolution)."""
    binding = mapping(clause.get("context_actor"))
    index, start, end = binding.get("evidence_index"), binding.get("start"), binding.get("end")
    citation = mapping(citations[index]) if type(index) is int and 0 <= index < len(citations) else {}
    cited, section = citation.get("sentence"), citation.get("section")
    if (j != 0 or clause.get("actor") is not None or not isinstance(cited, str) or not cited
            or not isinstance(section, str) or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(cited) or cited[start:end] != binding.get("text")
            or len(row) != 3 or not isinstance(row[0], str)):
        return False
    head_key = re.sub(r"^the ", "", entity_resolution_key(row[0]))
    if re.sub(r"^the ", "", entity_resolution_key(cited[start:end])) != head_key:
        return False
    before, rest = cited[:start], entity_resolution_key(cited[end:])
    text = entity_resolution_key(str(clause_text)) if (clause_text := row[2]) else ""
    if (before.strip() and not _SENTENCE_END.search(before)) or not text \
            or not (rest == text or rest.startswith(text + " ")):
        return False
    return any(cited.strip() in split_sections(source.text).get(section, "")
               for source in snapshot.sources)


def _checks(snapshot: AuditSnapshot) -> int:
    """Number of representations compared: adapter maps, dims, vocabulary, closed schema."""
    return 2 + int("rules.json" in snapshot.parsed) + int("model_dims.json" in snapshot.parsed) + int(snapshot.has_view(GraphView.STORED)) \
        + int(snapshot.has_view(GraphView.CONFORMANT)) \
        + int(isinstance(snapshot.metadata.get("extraction_coverage"), Mapping)
              and bool(snapshot.sources) and snapshot.has_view(GraphView.STORED))


def _issue_finding(ctx: RuleContext, issue: _Issue) -> Finding:
    return ctx.finding(issue.message, severity=issue.severity, affected=issue.affected,
                       locations=(issue.ref,), witness=issue.witness, discriminator=issue.discriminator)


# --- ART003 -----------------------------------------------------------------------
_PATTERNS = {
    (True, False, True): ("trained_quarantined", "trained although quarantine.json rejects it"),
    (True, False, False): ("trained_outside_deliverable",
                           "trained but neither in the conformant deliverable nor quarantined"),
    (True, None, True): ("trained_quarantined", "trained although quarantine.json rejects it"),
    (False, True, None): ("deliverable_not_trained", "in the conformant deliverable but not trained"),
    (False, True, False): ("deliverable_not_trained", "in the conformant deliverable but not trained"),
    (False, True, True): ("deliverable_and_quarantined", "both delivered and quarantined"),
    (True, True, True): ("deliverable_and_quarantined", "trained and delivered but also quarantined"),
}


@builtin(RuleSpec(
    "ART003", "2", "artifact", "Reconciliation of stored, conformant and quarantined views",
    Severity.WARNING,
    basis="set comparison of assertion ids across views; a stored triple whose normalized form "
          "is in the conformant view counts as reconciled"))
def art003(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    views = triple_views(snap)
    if len(views) < 2:
        return Outcome.unavailable("two of the stored, conformant and quarantined views")
    rows = {v: rows_by_assertion(snap, v) for v in views}
    normalized = {v: _keys(snap, rows, v) for v in (GraphView.STORED, GraphView.CONFORMANT)}
    reasons = quarantine_reasons(snap)
    ids = sorted({aid for r in rows.values() for aid in r})
    findings = [f for aid in ids if (f := _reconcile(ctx, aid, rows, normalized, reasons))]
    return Outcome(eligible=len(ids), findings=tuple(findings),
                   metrics=_view_sizes(rows, normalized))


def _keys(snapshot: AuditSnapshot, rows: dict[GraphView, dict[str, list[RawRef]]],
          view: GraphView) -> set[tuple[str, str, str]]:
    return {triple_key(*triple(snapshot, aid)) for aid in rows.get(view, {})}


def _membership_of(aid: str, rows: dict[GraphView, dict[str, list[RawRef]]]) -> tuple[bool | None, ...]:
    return tuple((aid in rows[v]) if v in rows else None
                 for v in (GraphView.STORED, GraphView.CONFORMANT, GraphView.QUARANTINED))


def _reconcile(ctx: RuleContext, aid: str, rows: dict[GraphView, dict[str, list[RawRef]]],
               normalized: dict[GraphView, set[tuple[str, str, str]]],
               reasons: dict) -> Finding | None:
    member = _membership_of(aid, rows)
    t = triple(ctx.snapshot, aid)
    if _respelled(member, triple_key(*t), normalized):
        return None  # the same fact under another spelling in the other view
    pattern = _PATTERNS.get(member)
    if pattern is None:
        return None
    key, text = pattern
    return ctx.finding(f"{list(t)} is {text}", affected=(aid,),
                       locations=tuple(ref for r in rows.values() for ref in r.get(aid, ())),
                       witness={"triple": list(t), "views": _present(member),
                                "quarantine_reasons": reasons.get(t, []),
                                "conformance_policy": ctx.snapshot.metadata.get("conformance_policy")},
                       discriminator=key, suggestion=_ART003_SUGGESTION[key])


_ART003_SUGGESTION = {
    "trained_quarantined": "review whether the quarantined triple should have been trained",
    "trained_outside_deliverable": "expected under the faithful conformance policy; review the "
                                   "out-of-vocabulary terms or extend the ontology",
    "deliverable_not_trained": "review why the deliverable fact is absent from the trained graph",
    "deliverable_and_quarantined": "the views contradict each other; regenerate the deliverable",
}


def _present(member: tuple[bool | None, ...]) -> dict[str, bool]:
    names = ("stored", "conformant", "quarantined")
    return {n: m for n, m in zip(names, member, strict=True) if m is not None}


def _respelled(member: tuple[bool | None, ...], key: tuple[str, str, str],
               normalized: dict[GraphView, set[tuple[str, str, str]]]) -> bool:
    stored, conformant, quarantined = member
    if quarantined:
        return False
    if stored and conformant is False:
        return key in normalized[GraphView.CONFORMANT]
    return bool(conformant and stored is False and key in normalized[GraphView.STORED])


def _view_sizes(rows: dict[GraphView, dict[str, list[RawRef]]],
                normalized: dict[GraphView, set[tuple[str, str, str]]]) -> dict[str, int]:
    return {f"{v.value}_assertions": len(r) for v, r in rows.items()} | {
        f"{v.value}_normalized_keys": len(k) for v, k in normalized.items() if v in rows}


# --- ART004 -----------------------------------------------------------------------
@builtin(RuleSpec(
    "ART004", "1", "artifact", "Repeated rows and repeated provenance", Severity.WARNING,
    basis="identical triples within one view's rows; repeated provenance records and identical "
          "evidence records; citing several passages is not a repetition"))
def art004(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    findings = [*_repeated_rows(ctx), *_repeated_provenance(ctx)]
    eligible = sum(len(snap.view(v)) for v in triple_views(snap)) + len(_provenance_records(snap))
    return Outcome(eligible=eligible, findings=tuple(findings))


def _repeated_rows(ctx: RuleContext) -> Iterator[Finding]:
    reasons = _record_reasons(ctx.snapshot)
    for view in triple_views(ctx.snapshot):
        for aid, refs in rows_by_assertion(ctx.snapshot, view).items():
            if len(refs) < 2:
                continue
            per_record = [reasons.get(r.pointer.removesuffix("/triple"), []) for r in refs]
            distinct = view is GraphView.QUARANTINED and len({tuple(p) for p in per_record}) > 1
            yield ctx.finding(
                f"{list(triple(ctx.snapshot, aid))} appears {len(refs)} times in the {view.value} view"
                + (" with different quarantine reasons" if distinct else ""),
                severity=Severity.INFO if distinct else Severity.WARNING, affected=(aid,),
                locations=tuple(refs), view=view, discriminator=f"rows:{view.value}",
                witness={"count": len(refs)} | ({"reasons_per_record": per_record} if distinct else {}))


def _record_reasons(snapshot: AuditSnapshot) -> dict[str, list[str]]:
    return {f"/{i}": [r for r in sequence(mapping(rec).get("reasons")) if isinstance(r, str)]
            for i, rec in enumerate(sequence(snapshot.parsed.get("quarantine.json")))}


def _provenance_records(snapshot: AuditSnapshot) -> tuple[FrozenJson, ...]:
    return sequence(artifact(snapshot, "provenance.json").get("triples"))


def _repeated_provenance(ctx: RuleContext) -> Iterator[Finding]:
    by_triple: dict[tuple, list[int]] = {}
    for i, rec in enumerate(_provenance_records(ctx.snapshot)):
        t = sequence(mapping(rec).get("triple"))
        by_triple.setdefault(tuple(t), []).append(i)
        yield from _repeated_evidence(ctx, i, mapping(rec))
    for t, indices in by_triple.items():
        if len(indices) > 1:
            yield ctx.finding(f"provenance.json has {len(indices)} records for {list(t)}",
                              locations=tuple(RawRef("provenance.json", f"/triples/{i}") for i in indices),
                              discriminator="provenance_record", witness={"triple": list(t)})


def _repeated_evidence(ctx: RuleContext, i: int, rec: Mapping[str, FrozenJson]) -> Iterator[Finding]:
    seen: dict[str, list[int]] = {}
    for j, ev in enumerate(sequence(rec.get("evidence"))):
        seen.setdefault(repr(thaw(ev)), []).append(j)
    for js in seen.values():
        if len(js) > 1:
            yield ctx.finding(f"provenance record {i} repeats one evidence record at indices {js}",
                              locations=tuple(RawRef("provenance.json", f"/triples/{i}/evidence/{j}")
                                              for j in js),
                              discriminator="evidence_record", witness={"count": len(js)})


# --- ART005 -----------------------------------------------------------------------
@builtin(RuleSpec(
    "ART005", "1", SNAPSHOT_CATEGORY, "Snapshot stability", Severity.ERROR,
    basis="file size and modification time before and after the read; a producer manifest, "
          "when OntoDisc writes one"))
def art005(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    changed = [d for d in snap.diagnostics if d.code in SNAPSHOT_CODES]
    findings = [ctx.finding(f"{d.location.artifact}: {d.message}", locations=(d.location,),
                            discriminator=d.code, witness={"code": d.code},
                            suggestion="re-run the audit when the store is not being written")
                for d in changed]
    present = sum(a.present for a in snap.artifacts)
    return Outcome(eligible=present, findings=tuple(findings),
                   metrics={"producer_manifest": "absent", "files_compared": present})


