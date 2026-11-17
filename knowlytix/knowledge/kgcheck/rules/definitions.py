# SPDX-License-Identifier: Apache-2.0
"""Definition rules DEF001-DEF003: missing, placeholder and duplicate descriptions;
description style; descriptions that do not name their parent.

Descriptions come only from the audit schema sidecar; OntoDisc stores record
none. The targets are declared terms (named by ``ontology.json`` or the sidecar)
of the configured kinds; a token that only occurs in triples is not a target.
All three checks are lexical. A description that mentions its own label is not
thereby circular, and one that names no parent is a suggestion to review, not a
logical contradiction.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from knowlytix.knowledge.kgcheck.audit_schema import SIDECAR
from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    Finding,
    Node,
    NodeKind,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import label_key
from knowlytix.knowledge.kgcheck.registry import (
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
    ratio,
)
from knowlytix.knowledge.kgcheck.rules._common import label
from knowlytix.knowledge.kgcheck.taxonomy import (
    HIERARCHY_VIEWS,
    SUBCLASS_PREDICATES,
    Hierarchy,
    accepted_predicates,
    hierarchy,
)

_KINDS = {"target_kinds": ("class", "property")}
_KIND_CHOICES = {"target_kinds": tuple(k.value for k in NodeKind)}
_PUNCTUATION = re.compile(r"[^\w\s]|_")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,5}s?\b")
_ROMAN = re.compile(r"[IVXLCDM]+")
# RFC 2119 requirement keywords are capitalized for emphasis, not acronyms
_REQUIREMENT_WORDS = ("MUST", "NOT", "SHALL", "SHOULD", "MAY", "REQUIRED", "RECOMMENDED",
                      "OPTIONAL")
_WITNESS = 50


def declared(snapshot: AuditSnapshot, node: Node) -> bool:
    """A term named by ontology.json's vocabularies or by the audit schema, not only by a row."""
    groups = snapshot.ontology.entities if node.namespace == "entity" else snapshot.ontology.relations
    return node.label in groups or SIDECAR in node.declared_by


def targets(snapshot: AuditSnapshot, kinds: tuple[str, ...]) -> list[Node]:
    """Declared terms of the given kinds, in node-id order.

    A described term whose kind is unknown is a target too: an unknown kind does
    not establish that the term is outside the configured kinds.
    """
    wanted = {NodeKind(k) for k in kinds}
    return [n for n in snapshot.nodes if declared(snapshot, n)
            and (n.kind in wanted or (n.kind is NodeKind.UNKNOWN and n.description is not None))]


def unassessed(snapshot: AuditSnapshot, kinds: tuple[str, ...]) -> int:
    """Declared terms of unknown kind without a description: not known to need one."""
    if NodeKind.UNKNOWN.value in kinds:
        return 0
    return sum(1 for n in snapshot.nodes if declared(snapshot, n)
               and n.kind is NodeKind.UNKNOWN and n.description is None)


def _plain(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", text).casefold().split())


def _compact(text: str) -> str:
    """Letters and digits only, so "N/A", "n.a." and "na" agree."""
    return "".join(_plain(text).split())


# --- DEF001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "DEF001", "2", "definitions", "Missing, placeholder and duplicate descriptions",
    Severity.WARNING, phase="B", prerequisites=("descriptions",), parameter_choices=_KIND_CHOICES,
    parameters={**_KINDS, "placeholders": ("tbd", "tba", "todo", "fixme", "na", "none", "null",
                                           "placeholder", "description", "definition", "xxx",
                                           "lorem ipsum", "to be defined", "to be determined")},
    basis="declared descriptions of the target kinds, and of described terms of unknown kind; "
          "placeholders compared ignoring case, punctuation and spacing; duplicates by identical "
          "normalized text"))
def def001(ctx: RuleContext) -> Outcome:
    nodes = targets(ctx.snapshot, tuple(ctx.params["target_kinds"]))
    placeholders = {_compact(str(p)) for p in ctx.params["placeholders"]}
    findings: list[Finding] = []
    texts: dict[str, list[Node]] = {}
    for node in nodes:
        text = _plain(node.description or "")
        if node.description is None:
            findings.append(_missing(ctx, node))
        elif not text or _compact(text) in placeholders:
            findings.append(_placeholder(ctx, node))
        else:
            texts.setdefault(text, []).append(node)
    findings += [_duplicate(ctx, group) for group in texts.values() if len(group) > 1]
    return Outcome(eligible=len(nodes), findings=tuple(findings), metrics={
        "coverage": {k: ratio(sum(n.description is not None for n in nodes if n.kind.value == k),
                              sum(n.kind.value == k for n in nodes))
                     for k in ctx.params["target_kinds"]},
        "unknown_kind_not_assessed": unassessed(ctx.snapshot, tuple(ctx.params["target_kinds"]))})


def _missing(ctx: RuleContext, node: Node) -> Finding:
    return ctx.finding(f"{node.kind.value} {node.label!r} has no description",
                       affected=(node.node_id,), discriminator="missing",
                       suggestion="add a description to the audit schema")


def _placeholder(ctx: RuleContext, node: Node) -> Finding:
    return ctx.finding(f"{node.kind.value} {node.label!r} has a placeholder description",
                       affected=(node.node_id,), witness={"description": node.description or ""},
                       discriminator="placeholder", suggestion="replace the placeholder")


def _duplicate(ctx: RuleContext, group: list[Node]) -> Finding:
    names = [n.label for n in group]
    return ctx.finding(f"{len(group)} terms share one description: {', '.join(map(repr, names[:5]))}",
                       affected=tuple(n.node_id for n in group),
                       witness={"labels": names[:_WITNESS], "description": group[0].description or ""},
                       discriminator="duplicate",
                       suggestion="distinguish the descriptions, or merge the terms if they are "
                                  "the same concept")


# --- DEF002 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "DEF002", "2", "definitions", "Description style", Severity.SUGGESTION, phase="B",
    prerequisites=("descriptions",),
    parameters={**_KINDS, "min_words": 3, "max_words": 120,
                "allowed_acronyms": _REQUIREMENT_WORDS,
                "discouraged_openings": ("this is", "this term", "refers to", "used to",
                                         "a term", "the term")},
    parameter_bounds={"min_words": (0, None), "max_words": (1, None)},
    parameter_order=(("min_words", "max_words"),), parameter_choices=_KIND_CHOICES,
    basis="word counts; a description equal to its label; upper-case acronyms (Roman numerals "
          "excepted) neither introduced in parentheses, declared as terms or aliases, nor allowed; "
          "configured openings"))
def def002(ctx: RuleContext) -> Outcome:
    nodes = [n for n in targets(ctx.snapshot, tuple(ctx.params["target_kinds"]))
             if n.description and _plain(n.description)]
    known = _known_acronyms(ctx)
    findings = [f for n in nodes for f in _style(ctx, n, known)]
    return Outcome(eligible=len(nodes), findings=tuple(findings))


def _known_acronyms(ctx: RuleContext) -> set[str]:
    """Allowed words and the labels and aliases of declared terms (not tokens only in rows)."""
    words = {str(w) for w in ctx.params["allowed_acronyms"]}
    for node in ctx.snapshot.nodes:
        if declared(ctx.snapshot, node):
            words |= {node.label, *node.aliases}
    return words


def _style(ctx: RuleContext, node: Node, known: set[str]) -> Iterator[Finding]:
    text = node.description or ""
    words = len(text.split())
    low, high = int(ctx.params["min_words"]), int(ctx.params["max_words"])
    if words < low or words > high:
        yield _styled(ctx, node, "length", f"has {words} words (expected {low}-{high})")
    if label_key(re.sub(r"^(?:an?|the)\s+|\.$", "", text.strip(), flags=re.I)) == label_key(node.label):
        yield _styled(ctx, node, "restates_label", "only restates the label")
    undefined = sorted({a for a in _ACRONYM.findall(text) if not _introduced(a, text, known)})
    if undefined:
        yield _styled(ctx, node, "acronym", f"uses acronyms it does not introduce: "
                      f"{', '.join(undefined)}", {"acronyms": undefined})
    opening = next((o for o in ctx.params["discouraged_openings"]
                    if _plain(text).startswith(_plain(str(o)) + " ")), None)
    if opening:
        yield _styled(ctx, node, "opening", f"opens with {opening!r}", {"opening": opening})


def _introduced(acronym: str, text: str, known: set[str]) -> bool:
    singular = acronym[:-1] if acronym.endswith("s") else acronym
    return (bool(_ROMAN.fullmatch(acronym)) or acronym in known or singular in known
            or f"({singular})" in text or f"({singular}s)" in text)


def _styled(ctx: RuleContext, node: Node, check: str, problem: str,
            witness: dict[str, object] | None = None) -> Finding:
    return ctx.finding(f"the description of {node.label!r} {problem}", affected=(node.node_id,),
                       witness={"description": (node.description or "")[:300], **(witness or {})},
                       discriminator=check, suggestion="revise the description")


# --- DEF003 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "DEF003", "3", "definitions", "Descriptions that do not name a parent", Severity.SUGGESTION,
    phase="B", prerequisites=("descriptions", "hierarchy"), views=HIERARCHY_VIEWS,
    basis="whether a described term's description contains the label or an alias of one of its "
          "parents (subclass parents in the view and declared types) as a whole-word sequence, "
          "singular or plural, punctuation ignored on both sides; genus and differentia are not "
          "assessed separately"))
def def003(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    h = hierarchy(snap, ctx.view)
    if not h.predicates and not h.types and not accepted_predicates(snap, DeclKey.TYPE_PREDICATE):
        return Outcome.unavailable(SUBCLASS_PREDICATES)
    described = [n for n in snap.nodes if n.description and _plain(n.description)]
    eligible = [(n, _parents(h, n.node_id)) for n in described]
    eligible = [(n, ps) for n, ps in eligible if ps]
    findings = [_no_genus(ctx, n, ps) for n, ps in eligible if not _names_any(snap, n, ps)]
    return Outcome(eligible=len(eligible), findings=tuple(findings),
                   metrics={"described": len(described), "genus_differentia": "not assessed"})


def _parents(h: Hierarchy, node_id: str) -> list[str]:
    return sorted({*h.parents.get(node_id, {}), *h.types.get(node_id, ())})


def _names_any(snapshot: AuditSnapshot, node: Node, parents: list[str]) -> bool:
    text = f" {label_key(_PUNCTUATION.sub(' ', node.description or ''))} "
    for parent in parents:
        p = snapshot.node_index.get(parent)
        for name in (label(snapshot, parent), *(p.aliases if p else ())):
            key = label_key(_PUNCTUATION.sub(" ", name))
            if key and any(f" {form} " in text for form in _forms(key)):
                return True
    return False


def _forms(key: str) -> tuple[str, ...]:
    """A label key with its regular English plurals."""
    plural = f"{key[:-1]}ies" if key.endswith("y") and key[-2:-1] not in "aeiou" else f"{key}s"
    return key, plural, f"{key}es"


def _no_genus(ctx: RuleContext, node: Node, parents: list[str]) -> Finding:
    names = [label(ctx.snapshot, p) for p in parents]
    return ctx.finding(f"the description of {node.label!r} names none of its parents: "
                       f"{', '.join(map(repr, names[:5]))}", affected=(node.node_id,),
                       witness={"description": (node.description or "")[:300], "parents": names},
                       discriminator="genus",
                       suggestion="state the parent class the term specializes, then how it "
                                  "differs")
