# SPDX-License-Identifier: Apache-2.0
"""EVD004: provenance against a rerun of OntoDisc's lexical grounding.

EVD001-EVD003 check that citations exist and point where they say. This rule
asks whether the current producer would still cite them: it reruns the
producer's provenance step (lexical grounding, then the structural derivation
of tables) on each cited section through ``knowlytix.discovery.ontodisc.audit``,
the only module that may run producer code. Grounding is a lexical heuristic:
a fact the rerun grounds may still be false, and one it does not ground may be
true. The rule reports agreement with the producer, not entailment.

The store does not record the grounding window. The window widens only the
context ``span`` of a citation, not whether a fact is grounded or which
sentence is cited, so spans are not compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType

from knowlytix.knowledge.kgcheck.frozen import thaw
from knowlytix.knowledge.kgcheck.models import (
    EvidenceLink,
    Finding,
    GraphView,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import (
    ONTOLOGY,
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
    ratio,
)
from knowlytix.knowledge.kgcheck.rules._common import rows_by_assertion, triple
from knowlytix.knowledge.kgcheck.sources import split_sections

RERUN_KINDS = frozenset({"sentence", "structure", "source_span"})  # what the rerun can produce


@builtin(RuleSpec(
    "EVD004", "1", "evidence", "Grounding against the producer", Severity.WARNING, phase="C",
    prerequisites=("provenance", "source_text", ONTOLOGY),
    views=(GraphView.STORED, GraphView.CONFORMANT),
    parameters={"grounding_window": 2}, parameter_bounds={"grounding_window": (1, 20)},
    cacheable=False,
    basis="reruns OntoDisc's provenance step (triple_evidence, then the structural derivation; "
          "knowlytix.discovery.ontodisc.audit) on each cited section with ontology.json; a "
          "lexical grounding heuristic, not semantic verification. Citations are compared by "
          "kind, sentence index and sentence; line citations cannot be rerun"))
def evd004(ctx: RuleContext) -> Outcome:
    from knowlytix.discovery.ontodisc import audit as producer
    snap = ctx.snapshot
    sections = split_sections(snap.sources[0].text)
    ontology = thaw(snap.parsed["ontology.json"])
    links: dict[tuple[str, str], list[EvidenceLink]] = {}
    for link in snap.evidence:
        if link.section in sections:  # a missing section is EVD002's finding
            links.setdefault((link.assertion_id, link.section), []).append(link)
    in_view = rows_by_assertion(snap, ctx.view)
    findings: list[Finding] = []
    counts = {"regrounded": 0, "citations_compared": 0, "citations_agreeing": 0,
              "not_rerunnable": 0, "producer_errors": 0}
    pairs = sorted(k for k in links if k[0] in in_view)
    for aid, section in pairs:
        findings += _check(ctx, producer, aid, section, sections[section], links[(aid, section)],
                           ontology, counts)
    tested = len(pairs) - counts["producer_errors"]
    return Outcome(
        eligible=len(pairs), tested=tested, findings=tuple(findings),
        partial=f"the producer raised on {counts['producer_errors']} citations"
        if counts["producer_errors"] else None,
        metrics={"heuristic": True, "grounding_window": ctx.params["grounding_window"],
                 "regrounded": ratio(counts["regrounded"], tested),
                 "citation_agreement": ratio(counts["citations_agreeing"],
                                             counts["citations_compared"]),
                 "not_rerunnable": counts["not_rerunnable"],
                 "producer_errors": counts["producer_errors"]})


def _check(ctx: RuleContext, producer: ModuleType, aid: str, section: str, text: str,
           recorded: list[EvidenceLink], ontology: Mapping, counts: dict[str, int]) -> list[Finding]:
    h, r, o = triple(ctx.snapshot, aid)
    try:
        rerun = producer.expected_evidence(h, r, o, section, text, ontology,
                                           int(ctx.params["grounding_window"]))
    except Exception:  # noqa: BLE001  # a producer failure makes this citation untested
        counts["producer_errors"] += 1
        return []
    if not rerun:
        return [ctx.finding(
            f"the producer no longer grounds {[h, r, o]} in section {section!r}",
            affected=(aid,), locations=tuple(lk.raw for lk in recorded[:5]),
            discriminator=f"not_regrounded:{section}",
            witness={"triple": [h, r, o], "section": section,
                     "recorded": [_cite(lk) for lk in recorded[:5]]},
            suggestion="review the fact against the section; rebuild provenance if the "
                       "producer changed")]
    counts["regrounded"] += 1
    expected = {(e.get("kind"), e.get("sentence_index"), e.get("sentence")) for e in rerun}
    findings = []
    for link in recorded:
        if link.kind not in RERUN_KINDS:
            counts["not_rerunnable"] += 1
            continue
        counts["citations_compared"] += 1
        if (link.kind, link.sentence_index, link.sentence) in expected:
            counts["citations_agreeing"] += 1
            continue
        findings.append(ctx.finding(
            f"the producer grounds {[h, r, o]} in section {section!r} but cites other text",
            severity=Severity.INFO, affected=(aid,), locations=(link.raw,),
            discriminator=f"citation_differs:{link.raw.pointer}",
            witness={"recorded": _cite(link), "rerun": [_short(e) for e in rerun[:5]]},
            suggestion="rebuild provenance to cite what the current producer cites"))
    return findings


def _cite(link: EvidenceLink) -> dict[str, object]:
    return {"kind": link.kind, "sentence_index": link.sentence_index,
            "sentence": link.sentence[:300]}


def _short(e: Mapping) -> dict[str, object]:
    return {"kind": e.get("kind"), "sentence_index": e.get("sentence_index"),
            "sentence": str(e.get("sentence", ""))[:300]}
