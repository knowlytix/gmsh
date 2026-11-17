# SPDX-License-Identifier: Apache-2.0
"""ART006: rule records against a rerun of the OntoDisc clause grammar.

Every other rule checks artifacts with kgcheck's own reimplementation of the
producer's conventions. Two checks of ``rules.json`` depend on the producer's
clause grammar itself: whether a clause labelled descriptive is what the grammar
parses at its span, and whether a context actor is the one the producer
resolves from the cited sentences. Reimplementing that grammar would copy it,
so this rule reruns it through ``knowlytix.discovery.ontodisc.audit``. A match
shows that the record agrees with the current producer; a defect in the
grammar itself would be reproduced, not detected. ART002 checks the same
records independently: spans, readiness and the exact source of a context actor.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import ModuleType

from knowlytix.knowledge.kgcheck.frozen import FrozenJson, thaw
from knowlytix.knowledge.kgcheck.models import Finding, RawRef, Severity
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import artifact, mapping, sequence

_FIELDS = ("actor", "action", "modality", "deontic")


@builtin(RuleSpec(
    "ART006", "1", "artifact", "Rule records against the producer's clause grammar",
    Severity.ERROR, phase="B", prerequisites=("artifact:rules.json",), cacheable=False,
    basis="reruns OntoDisc's parse_clauses and source_actor (knowlytix.discovery.ontodisc.audit) "
          "on each rule's text; agreement means the record matches the current producer, not "
          "that the producer's grammar is correct"))
def art006(ctx: RuleContext) -> Outcome:
    from knowlytix.discovery.ontodisc import audit as producer
    checked, findings = 0, []
    for i, raw in enumerate(sequence(artifact(ctx.snapshot, "rules.json").get("rules"))):
        record = mapping(raw)
        text, identity = record.get("text"), record.get("id")
        if not isinstance(text, str):
            continue
        for j, raw_clause in enumerate(sequence(record.get("clauses"))):
            clause = mapping(raw_clause)
            ref = RawRef("rules.json", f"/rules/{i}/clauses/{j}")
            if clause.get("kind") == "descriptive":
                checked += 1
                findings += _descriptive(ctx, producer, clause, text, str(identity), ref, i, j)
            if clause.get("context_actor") is not None:
                checked += 1
                findings += _actor(ctx, producer, record, clause, text, ref, i, j)
    return Outcome(eligible=checked, findings=tuple(findings),
                   metrics={"producer_reruns": checked})


def _descriptive(ctx: RuleContext, producer: ModuleType, clause: Mapping[str, FrozenJson],
                 text: str, identity: str, ref: RawRef, i: int, j: int) -> Iterator[Finding]:
    span = thaw(clause.get("span"))
    start, end = (span.get("start"), span.get("end")) if isinstance(span, dict) else (None, None)
    expected = (producer.expected_clause(text, identity, span)
                if type(start) is int and type(end) is int and 0 <= start < end <= len(text) else None)
    if (expected is None or expected.get("kind") != "descriptive"
            or any(thaw(clause.get(k)) != expected.get(k) for k in _FIELDS)):
        yield ctx.finding("descriptive clause does not match the producer's clause grammar",
                          locations=(ref,), discriminator=f"clause_description:{i}:{j}",
                          witness={"expected": {k: expected.get(k) for k in ("kind", *_FIELDS)}
                                   if expected else None})


def _actor(ctx: RuleContext, producer: ModuleType, record: Mapping[str, FrozenJson],
           clause: Mapping[str, FrozenJson], text: str, ref: RawRef, i: int, j: int
           ) -> Iterator[Finding]:
    row = sequence(record.get("triple"))
    head = row[0] if len(row) == 3 and isinstance(row[0], str) else None
    evidence = thaw(record.get("evidence"))
    expected = (producer.expected_context_actor(text, head, evidence if isinstance(evidence, list) else [])
                if head is not None else None)
    if thaw(clause.get("context_actor")) != expected:
        yield ctx.finding("context actor differs from the producer's resolution",
                          locations=(ref,), discriminator=f"context_actor_grammar:{i}:{j}",
                          witness={"recorded": thaw(clause.get("context_actor")),
                                   "expected": expected})
