# SPDX-License-Identifier: Apache-2.0
"""Coverage rules: which source sections the view's facts cite (COV001), and the
answers the view gives to declared competency questions (COV002).

OntoDisc stores do not persist which sections extraction attempted, which
failed, or which were held out for confirmation, so those denominators are
reported as unknown rather than inferred. Competency questions measure
coverage of a declared task on a fixed graph, not completeness in general.
"""

from __future__ import annotations

from knowlytix.knowledge.kgcheck.adapters._json import pointer_token
from knowlytix.knowledge.kgcheck.datatypes import is_number, value_key
from knowlytix.knowledge.kgcheck.models import (
    CompetencyQuestion,
    GraphView,
    RawRef,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import (
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
    ratio,
)
from knowlytix.knowledge.kgcheck.rules._common import (
    Resolver,
    entity_resolver,
    relation_resolver,
    rows_by_assertion,
    triple,
)
from knowlytix.knowledge.kgcheck.sources import split_sections

UNKNOWN = ("sections attempted by extraction", "sections whose extraction failed",
           "held-out versus training sections")


@builtin(RuleSpec(
    "COV001", "1", "coverage", "Source sections cited by the view", Severity.INFO,
    prerequisites=("provenance", "source_text"), views=(GraphView.STORED, GraphView.CONFORMANT),
    basis="sections of documents/combined.md (OntoDisc section keys) against the sections cited "
          "by evidence for the view's assertions"))
def cov001(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    sections = split_sections(snap.sources[0].text)
    in_view = set(rows_by_assertion(snap, ctx.view))
    cited = {link.section for link in snap.evidence if link.assertion_id in in_view}
    uncited = [s for s in sections if s not in cited]
    source = snap.sources[0].artifact
    findings = tuple(ctx.finding(f"section {name!r} has no cited {ctx.view.value} fact",
                                 locations=(RawRef(source, f"/sections/{pointer_token(name)}"),),
                                 witness={"section": name, "characters": len(sections[name])},
                                 discriminator=name,
                                 suggestion="check whether extraction covered this section")
                     for name in uncited)
    return Outcome(eligible=len(sections), findings=findings, metrics={
        "sections_with_cited_facts": ratio(len(sections) - len(uncited), len(sections)),
        "cited_sections_absent_from_source": sorted(cited - set(sections)),
        "unknown": list(UNKNOWN)})


# --- COV002 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "COV002", "2", "coverage", "Competency questions", Severity.WARNING, phase="C",
    prerequisites=("competency_questions",), views=(GraphView.STORED, GraphView.CONFORMANT),
    basis="each declared competency question answered from the view's triples: terms match "
          "when they resolve to the same declared term through the ontology's names and "
          "aliases (kgcheck resolution keys), numbers by value; a missing expected answer, no "
          "answer to a question that lists none, an unexpected one for an exhaustive question, "
          "or a stated fact asked to be absent is a finding"))
def cov002(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    questions = snap.semantics.accepted().competency
    if not questions:
        return Outcome.unavailable("competency_questions")
    terms = _Terms(entity_resolver(snap), relation_resolver(snap))
    facts = [triple(snap, aid) for aid in rows_by_assertion(snap, ctx.view)]
    findings, answered, expected, found = [], 0, 0, 0
    for q in questions:
        result = _answer(q, facts, terms)
        expected += result.expected
        found += result.found
        answered += not result.findings
        findings += [ctx.finding(message, affected=(f"question:{q.question_id}",),
                                 locations=(q.raw,), discriminator=code,
                                 witness={"question": q.question or None, "relation": q.relation,
                                          "head": q.head, "object": q.object,
                                          "expected": list(q.expected), "observed": result.observed},
                                 suggestion="check extraction of the facts the question needs")
                     for code, message in result.findings]
    return Outcome(eligible=len(questions), findings=tuple(findings), metrics={
        "questions_answered": ratio(answered, len(questions)),
        "expected_answers_found": ratio(found, expected)})


class _Terms:
    """Comparison keys for question terms and stored tokens."""

    def __init__(self, entities: Resolver, relations: Resolver) -> None:
        self.entities, self.relations = entities, relations

    def entity(self, token: str) -> str:
        if is_number(token):
            return value_key(token)
        return _key(self.entities, token)

    def relation(self, token: str) -> str:
        return _key(self.relations, token)


def _key(resolver: Resolver, token: str) -> str:
    targets = resolver.targets(token)
    if targets:
        return "term:" + targets[0]
    return "key:" + "|".join(sorted(resolver.keys(token)) or [token])


class _Answer:
    def __init__(self) -> None:
        self.observed: list[str] = []
        self.findings: list[tuple[str, str]] = []
        self.expected = 0
        self.found = 0


def _answer(q: CompetencyQuestion, facts: list[tuple[str, str, str]], terms: _Terms) -> _Answer:
    out, relation = _Answer(), terms.relation(q.relation)
    head = terms.entity(q.head) if q.head is not None else None
    obj = terms.entity(q.object) if q.object is not None else None
    matching = [(h, o) for h, r, o in facts if terms.relation(r) == relation
                and (head is None or terms.entity(h) == head) and (obj is None or terms.entity(o) == obj)]
    if head is not None and obj is not None:
        out.observed = ["stated"] if matching else []
        stated = bool(matching)
        if stated == q.absent:
            out.findings.append(("closed", f"question {q.question_id!r}: the view "
                                 f"{'states' if stated else 'does not state'} "
                                 f"[{q.head!r}, {q.relation!r}, {q.object!r}]"))
        return out
    answers = sorted({o if head is not None else h for h, o in matching})
    out.observed = answers[:50]
    observed_keys = {terms.entity(a) for a in answers}
    expected_keys = {terms.entity(e): e for e in q.expected}
    out.expected, missing = len(expected_keys), [e for k, e in expected_keys.items()
                                                 if k not in observed_keys]
    out.found = out.expected - len(missing)
    if missing:
        out.findings.append(("missing", f"question {q.question_id!r}: expected answers missing "
                             f"from the view: {', '.join(map(repr, missing))}"))
    extra = [a for a in answers if terms.entity(a) not in expected_keys]
    if q.exhaustive and extra:
        out.findings.append(("unexpected", f"question {q.question_id!r}: answers outside the "
                             f"expected set: {', '.join(map(repr, extra[:20]))}"))
    if q.absent and answers:
        out.findings.append(("present", f"question {q.question_id!r} expects no answer; the view "
                             f"gives {', '.join(map(repr, answers[:20]))}"))
    if not q.absent and not q.expected and not answers:
        out.findings.append(("unanswered", f"question {q.question_id!r} gets no answer from the "
                             "view"))
    return out
