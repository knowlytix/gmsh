# SPDX-License-Identifier: Apache-2.0
"""Evidence rules EVD001-EVD003: provenance coverage, citation validity, evidence keys.

These rules check that citations exist and point where they say. They do not
check that a cited passage supports its fact; that is grounding (EVD004, a
later phase) and entailment, neither of which an artifact audit performs.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Mapping
from decimal import Decimal, InvalidOperation
from functools import cached_property

from knowlytix.knowledge.kgcheck.frozen import FrozenJson, thaw
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    EvidenceLink,
    Finding,
    GraphView,
    RawRef,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import (
    DATE_SURFACE,
    is_numeric_token,
    relation_resolution_key,
    triple_key,
)
from knowlytix.knowledge.kgcheck.registry import (
    ONTOLOGY,
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
    ratio,
)
from knowlytix.knowledge.kgcheck.rules._common import (
    Triple,
    artifact,
    entity_resolver,
    mapping,
    relation_resolver,
    rows_by_assertion,
    sequence,
    triple,
    triple_views,
)
from knowlytix.knowledge.kgcheck.sources import (
    MONEY_RELATIONS,
    monetary_units,
    nonblank_lines,
    sentence_units,
    split_sections,
)

_EVIDENCE_VIEWS = (GraphView.STORED, GraphView.CONFORMANT)
# A numeral as OntoDisc's canonical_number reads it: comma groups of three, an
# optional "(...)" negative and a magnitude word. Digits inside a date are not numerals.
_NUMERAL = re.compile(
    r"(?<![\w.])(?P<open>\$?\()?(?P<sign>[-+])?\$?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?"
    r"(?!\d)(?P<close>\))?(?:\s*(?P<mag>thousand|million|billion|bn|k|m|b)\b)?", re.I)
_MAGNITUDE = {"thousand": 1_000, "k": 1_000, "million": 1_000_000, "m": 1_000_000,
              "billion": 1_000_000_000, "bn": 1_000_000_000, "b": 1_000_000_000}
_AMBIGUOUS_MAGNITUDE = frozenset({"m", "b"})  # also units (metres, bytes): keep the bare number
SECTION_LEVEL = frozenset({"structure", "source_span"})


def _records(snapshot: AuditSnapshot) -> tuple[FrozenJson, ...]:
    return sequence(artifact(snapshot, "provenance.json").get("triples"))


def _record_triple(record: FrozenJson) -> Triple | None:
    t = sequence(mapping(record).get("triple"))
    return (t[0], t[1], t[2]) if len(t) == 3 and all(isinstance(x, str) for x in t) else None


# --- EVD001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "EVD001", "1", "evidence", "Provenance coverage of the view", Severity.WARNING,
    prerequisites=("provenance",), views=_EVIDENCE_VIEWS, parameters={"require_evidence": False},
    basis="left join of the view's assertions to provenance.json records by exact triple; a "
          "record with no valid evidence entry counts as missing. Coverage is not truth."))
def evd001(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    in_view = rows_by_assertion(snap, ctx.view)
    cited = {link.assertion_id for link in snap.evidence}
    severity = Severity.ERROR if ctx.params["require_evidence"] else Severity.WARNING
    findings = [_uncited(ctx, aid, refs, severity) for aid, refs in in_view.items() if aid not in cited]
    findings += list(_orphans(ctx, set(in_view)))
    return Outcome(eligible=len(in_view), findings=tuple(findings), metrics={
        "evidence_coverage": ratio(len(set(in_view) & cited), len(in_view))})


def _uncited(ctx: RuleContext, aid: str, refs: list[RawRef], severity: Severity) -> Finding:
    return ctx.finding(f"{list(triple(ctx.snapshot, aid))} has no evidence in provenance.json",
                       severity=severity, affected=(aid,), locations=tuple(refs[:1]),
                       discriminator="missing", suggestion="re-ground the fact or remove it")


def _orphans(ctx: RuleContext, in_view: set[str]) -> Iterator[Finding]:
    from knowlytix.knowledge.kgcheck.identity import assertion_id
    anywhere = {aid for v in triple_views(ctx.snapshot) for aid in rows_by_assertion(ctx.snapshot, v)}
    for i, record in enumerate(_records(ctx.snapshot)):
        t = _record_triple(record)
        if t is None or (aid := assertion_id(*t)) in in_view:
            continue
        absent = aid not in anywhere
        yield ctx.finding(f"provenance record for {list(t)} cites a triple "
                          f"{'in no view' if absent else f'outside the {ctx.view.value} view'}",
                          severity=Severity.WARNING if absent else Severity.INFO,
                          locations=(RawRef("provenance.json", f"/triples/{i}"),),
                          witness={"triple": list(t)}, discriminator="orphan")


# --- EVD002 -------------------------------------------------------------------------
class _Sources:
    """Sections of the retained source and the documents metadata lists."""

    def __init__(self, snapshot: AuditSnapshot) -> None:
        self.snapshot = snapshot

    @cached_property
    def sections(self) -> dict[str, str]:
        return split_sections(self.snapshot.sources[0].text) if self.snapshot.sources else {}

    @cached_property
    def documents(self) -> set[str] | None:
        docs = self.snapshot.metadata.get("documents")
        if not isinstance(docs, tuple):
            return None
        return {d["name"] for d in docs if isinstance(d, Mapping) and isinstance(d.get("name"), str)}

    def knows(self, document: str, section: str) -> bool:
        """The cited document is listed in metadata, or names the cited section.

        A multi-document build retains one combined source and names each
        section ``"<file name>: <heading>"`` (the full path when two files
        share a name), so the section key identifies the document.
        """
        if self.documents is None or document in self.documents:
            return True
        name = document.replace("\\", "/").rsplit("/", 1)[-1]
        return section.startswith((f"{name}: ", f"{document}: "))


@builtin(RuleSpec(
    "EVD002", "2", "evidence", "Validity of evidence citations", Severity.ERROR,
    prerequisites=("provenance", "source_text"),
    basis="each citation resolved in documents/combined.md: the section by OntoDisc's section "
          "keys, a sentence record by its index into the section's sentence units (monetary "
          "relations by OntoDisc's colon-preserving units), a line record by its index into "
          "non-blank lines; numeric objects are looked for among the numerals of the cited unit, "
          "or of the whole section for a section-level citation"))
def evd002(ctx: RuleContext) -> Outcome:
    sources = _Sources(ctx.snapshot)
    triples = _link_triples(ctx.snapshot)
    findings = [f for link in ctx.snapshot.evidence for f in _check_link(ctx, sources, link, triples)]
    findings += _section_level(ctx)
    kinds: dict[str, int] = {}
    for link in ctx.snapshot.evidence:
        kinds[link.kind] = kinds.get(link.kind, 0) + 1
    return Outcome(eligible=len(ctx.snapshot.evidence), findings=tuple(findings),
                   metrics={"citations_by_kind": dict(sorted(kinds.items())),
                            "sections_in_source": len(sources.sections)})


def _link_triples(snapshot: AuditSnapshot) -> dict[str, Triple]:
    """Each provenance record's triple, by the record pointer its links carry."""
    return {f"/triples/{i}": t for i, rec in enumerate(_records(snapshot))
            if (t := _record_triple(rec)) is not None}


def _record_of(link: EvidenceLink) -> str:
    return link.raw.pointer.rsplit("/evidence/", 1)[0]


def _check_link(ctx: RuleContext, sources: _Sources, link: EvidenceLink,
                triples: dict[str, Triple]) -> Iterator[Finding]:
    t = triples.get(_record_of(link))
    if link.document is not None and not sources.knows(link.document, link.section):
        yield _link_finding(ctx, link, t, "document_unknown",
                            f"cites document {link.document!r}, which metadata.json does not list")
    section = sources.sections.get(link.section)
    if section is None:
        yield _link_finding(ctx, link, t, "section_missing",
                            f"cites section {link.section!r}, absent from the retained source")
        return
    check = _UNIT_CHECKS.get(link.kind)
    if check is not None:  # an unknown kind is a contract violation (ART001)
        yield from check(ctx, link, t, section)


def _sentence(ctx: RuleContext, link: EvidenceLink, t: Triple | None, section: str) -> Iterator[Finding]:
    monetary = t is not None and relation_resolution_key(t[1]) in MONEY_RELATIONS
    units = [u.strip() for u in (monetary_units(section) if monetary else sentence_units(section))]
    yield from _unit(ctx, link, t, units, link.sentence, "sentence")
    if link.span and link.span not in " ".join(units):
        yield _link_finding(ctx, link, t, "span_not_in_section", "its span is not a passage of the "
                            "cited section", Severity.WARNING)


def _line(ctx: RuleContext, link: EvidenceLink, t: Triple | None, section: str) -> Iterator[Finding]:
    yield from _unit(ctx, link, t, [ln.strip() for ln in nonblank_lines(section)], link.span, "line")


def _unit(ctx: RuleContext, link: EvidenceLink, t: Triple | None, units: list[str], text: str,
          unit: str) -> Iterator[Finding]:
    i = link.sentence_index
    if i < len(units) and units[i] == text:
        yield from _value_check(ctx, link, t, text)
    elif text in units:
        yield _link_finding(ctx, link, t, "index_mismatch", f"{unit} index {i} does not point at "
                            f"the cited {unit}; it is {unit} {units.index(text)}", Severity.WARNING)
    else:
        yield _link_finding(ctx, link, t, "text_not_in_section",
                            f"the cited {unit} is not a {unit} of section {link.section!r}")


def _whole_section(ctx: RuleContext, link: EvidenceLink, t: Triple | None,
                   section: str) -> Iterator[Finding]:
    if link.sentence != section or link.span != section:
        yield _link_finding(ctx, link, t, "text_not_in_section", f"a {link.kind} record must "
                            f"carry the whole text of section {link.section!r}")
    else:
        yield from _value_check(ctx, link, t, section)


_UNIT_CHECKS = {"sentence": _sentence, "line": _line, "structure": _whole_section,
                "source_span": _whole_section}


def _value_check(ctx: RuleContext, link: EvidenceLink, t: Triple | None, text: str) -> Iterator[Finding]:
    if t is None or not is_numeric_token(t[2]):
        return
    numerals = _numerals(text)
    if Decimal(t[2]) not in numerals:
        stated = sorted(numerals)
        finding = _link_finding(ctx, link, t, "value_not_in_citation",
                                f"the cited {link.kind} states no number equal to {t[2]}",
                                Severity.WARNING)
        yield dataclasses.replace(finding, witness={**finding.witness, "numerals_in_citation": [
            str(n) for n in stated[:20]]})


def _numerals(text: str) -> set[Decimal]:
    """The numbers ``text`` states, read the way OntoDisc canonicalizes a number."""
    out: set[Decimal] = set()
    for m in _NUMERAL.finditer(DATE_SURFACE.sub(" ", text)):
        try:
            value = Decimal(m["int"].replace(",", "") + (m["frac"] or ""))
        except InvalidOperation:
            continue
        if m["sign"] == "-" or (m["open"] and m["close"]):
            value = -value
        out |= _scaled(value, (m["mag"] or "").lower())
    return out


def _scaled(value: Decimal, magnitude: str) -> set[Decimal]:
    if not magnitude:
        return {value}
    scaled = value * _MAGNITUDE[magnitude]
    return {scaled, value} if magnitude in _AMBIGUOUS_MAGNITUDE else {scaled}


def _link_finding(ctx: RuleContext, link: EvidenceLink, t: Triple | None, code: str, message: str,
                  severity: Severity | None = None) -> Finding:
    return ctx.finding(f"evidence for {list(t) if t else link.assertion_id} {message}",
                       severity=severity, affected=(link.assertion_id,), locations=(link.raw,),
                       witness={"kind": link.kind, "section": link.section,
                                "sentence_index": link.sentence_index, "cited": link.sentence[:300]},
                       discriminator=f"{code}:{link.raw.pointer}", suggestion=_EVD002_SUGGESTION[code])


_EVD002_SUGGESTION = {
    "document_unknown": "record the document in metadata.json or correct the citation",
    "section_missing": "the retained source and provenance disagree; rebuild provenance",
    "text_not_in_section": "the citation does not match the retained source; rebuild provenance",
    "index_mismatch": "the cited text exists at another index; rebuild provenance",
    "span_not_in_section": "the context span does not match the retained source",
    "value_not_in_citation": "review whether this passage states the value; a neighboring row or "
                             "sentence may have been cited",
    "section_level": "section-level citations identify where, not which sentence; review the "
                     "fact against the section",
}


def _section_level(ctx: RuleContext) -> list[Finding]:
    by_assertion: dict[str, list[EvidenceLink]] = {}
    for link in ctx.snapshot.evidence:
        if link.kind in SECTION_LEVEL:
            by_assertion.setdefault(link.assertion_id, []).append(link)
    triples = _link_triples(ctx.snapshot)
    return [ctx.finding(f"evidence for {list(t) if (t := triples.get(_record_of(links[0]))) else aid} "
                        f"cites {len(links)} whole section(s), not a sentence",
                        severity=Severity.INFO, affected=(aid,), locations=tuple(lk.raw for lk in links),
                        witness={"kinds": sorted({lk.kind for lk in links}),
                                 "sections": sorted({lk.section for lk in links})},
                        discriminator="section_level", suggestion=_EVD002_SUGGESTION["section_level"])
            for aid, links in by_assertion.items()]


# --- EVD003 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "EVD003", "2", "evidence", "Provenance ontology and evidence keys", Severity.WARNING,
    prerequisites=("provenance", ONTOLOGY),
    basis="the ontology embedded in provenance.json compared with ontology.json; each record's "
          "triple compared with view triples exactly and after kgcheck normalization"))
def evd003(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    findings = list(_stale_ontology(ctx)) + list(_evidence_keys(ctx))
    return Outcome(eligible=1 + len(_records(snap)), findings=tuple(findings))


def _stale_ontology(ctx: RuleContext) -> Iterator[Finding]:
    embedded = mapping(artifact(ctx.snapshot, "provenance.json").get("ontology"))
    declared = artifact(ctx.snapshot, "ontology.json")
    if thaw(embedded) == thaw(declared):
        return
    diff = {key: _group_diff(mapping(embedded.get(key)), mapping(declared.get(key)))
            for key in ("entities", "relations")}
    yield ctx.finding("the ontology embedded in provenance.json differs from ontology.json",
                      locations=(RawRef("provenance.json", "/ontology"), RawRef("ontology.json")),
                      witness={"difference": diff}, discriminator="stale_ontology",
                      suggestion="evidence was keyed under another ontology; rebuild provenance")


def _group_diff(embedded: Mapping[str, FrozenJson], declared: Mapping[str, FrozenJson]) -> dict:
    changed = sorted(k for k in set(embedded) & set(declared) if thaw(embedded[k]) != thaw(declared[k]))
    return {"only_in_provenance": sorted(set(embedded) - set(declared)),
            "only_in_ontology": sorted(set(declared) - set(embedded)), "aliases_differ": changed}


def _evidence_keys(ctx: RuleContext) -> Iterator[Finding]:
    snap = ctx.snapshot
    exact = {triple(snap, aid) for v in triple_views(snap) for aid in rows_by_assertion(snap, v)}
    normalized = {triple_key(*t): t for t in exact}
    ent, rel = entity_resolver(snap), relation_resolver(snap)
    for i, record in enumerate(_records(snap)):
        t = _record_triple(record)
        if t is None or t in exact:
            continue
        match = normalized.get(triple_key(*t))
        if match is None:
            continue  # evidence for a triple in no view is EVD001's orphan
        yield ctx.finding(f"provenance keys {list(t)} under a spelling no view uses; the view has "
                          f"{list(match)}", locations=(RawRef("provenance.json", f"/triples/{i}/triple"),),
                          witness={"triple": list(t), "view_spelling": list(match),
                                   "declared_head": ent.targets(t[0])[:1],
                                   "declared_relation": rel.targets(t[1])[:1]},
                          discriminator="respelled_key",
                          suggestion="evidence lookups by the stored spelling will miss this record")
