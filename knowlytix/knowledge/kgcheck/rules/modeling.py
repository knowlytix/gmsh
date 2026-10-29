# SPDX-License-Identifier: Apache-2.0
"""Modeling rules MOD001-MOD002: concepts named like properties, and properties
attributed to no concept.

MOD001 reads the last word of each declared class label against a list of
property-like words. The list is language specific: the shipped one is English,
and a class whose declared language the list does not cover is not checked.
MOD002 checks that each relation belongs to a concept under a chosen modeling
convention: ``schema_rows`` reads OntoDisc's ``schema`` rows ``(head type,
relation, tail type)`` as attribution, ``typed_heads`` accepts a relation used
on a declared class or typed instance. Both are review aids, not constraints.
"""

from __future__ import annotations

import re

from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    Finding,
    Node,
    NodeKind,
    SemanticsDecl,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import label_key
from knowlytix.knowledge.kgcheck.registry import (
    ONTOLOGY,
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
)
from knowlytix.knowledge.kgcheck.rules._common import (
    entity_resolver,
    label,
    relation_resolver,
    term_ref,
)
from knowlytix.knowledge.kgcheck.taxonomy import (
    Hierarchy,
    accepted_predicates,
    hierarchy,
)

_ENGLISH_SUFFIXES = ("amount", "count", "date", "rate", "name", "number", "id", "code", "value",
                     "percentage", "total", "status", "level", "size", "duration")
_WITNESS = 20
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# --- MOD001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "MOD001", "2", "modeling", "Concepts named like properties", Severity.SUGGESTION, phase="B",
    prerequisites=("node_kinds",),
    parameters={"property_suffixes": _ENGLISH_SUFFIXES, "languages": ("en",)},
    basis="the closing words of each declared class label (camelCase split, kgcheck label key) "
          "compared with property-like words for the class's declared language (primary subtag, "
          "any case); classes in other languages are listed as not checked"))
def mod001(ctx: RuleContext) -> Outcome:
    sem = ctx.snapshot.semantics.accepted()
    classes = [n for n in ctx.snapshot.nodes if n.kind is NodeKind.CLASS]
    languages = {n.node_id: _language(sem, n.node_id) for n in classes}
    if classes and not any(languages.values()):
        return Outcome.unavailable("language")
    covered = {_primary(str(x)) for x in ctx.params["languages"]}
    checked = [n for n in classes if languages[n.node_id] in covered]
    skipped = [n for n in classes if languages[n.node_id] not in covered]
    unsupported = sorted({lang for n in skipped if (lang := languages[n.node_id])})
    if classes and not checked:
        return Outcome.unavailable(*(f"property_suffixes:{lang}" for lang in unsupported))
    suffixes = [w for s in ctx.params["property_suffixes"] if (w := _words(str(s)))]
    findings = [_property_like(ctx, n, " ".join(match)) for n in checked
                if (match := _suffix(_words(n.label), suffixes))]
    if skipped:
        findings.append(_unchecked(ctx, skipped, languages))
    return Outcome(eligible=len(classes), findings=tuple(findings),
                   metrics={"classes": len(classes), "checked": len(checked),
                            "unchecked_languages": unsupported,
                            "without_language": sum(not v for v in languages.values())})


def _language(sem: SemanticsDecl, node_id: str) -> str | None:
    own = sem.find(DeclKey.LANGUAGE, node_id) or sem.find(DeclKey.LANGUAGE, "")
    return _primary(own.value) if own else None


def _primary(tag: str) -> str:
    """The primary language subtag of a BCP 47 tag, lower case: "en-US" -> "en"."""
    return tag.replace("_", "-").split("-")[0].casefold()


def _words(label: str) -> list[str]:
    return label_key(_CAMEL.sub(" ", label)).replace(".", " ").split()


def _suffix(words: list[str], suffixes: list[list[str]]) -> list[str] | None:
    """The longest configured suffix the label ends with."""
    ends = [s for s in suffixes if len(s) <= len(words) and words[-len(s):] == s]
    return max(ends, key=len) if ends else None


def _unchecked(ctx: RuleContext, skipped: list[Node], languages: dict[str, str | None]) -> Finding:
    return ctx.finding(f"{len(skipped)} classes were not checked: no declared language the word "
                       "list covers", severity=Severity.INFO, discriminator="unchecked",
                       affected=tuple(n.node_id for n in skipped),
                       witness={"classes": {n.label: languages[n.node_id] for n in skipped[:_WITNESS]}},
                       suggestion="declare their language, or supply property_suffixes for it")


def _property_like(ctx: RuleContext, node: Node, word: str) -> Finding:
    return ctx.finding(f"class {node.label!r} ends in {word!r}, which usually names a property",
                       affected=(node.node_id,), witness={"label": node.label, "word": word},
                       suggestion="review: model the value as a relation of the concept it "
                                  "describes, unless the class is a concept in its own right",
                       discriminator="property_like")


# --- MOD002 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "MOD002", "2", "modeling", "Properties attributed to no concept", Severity.WARNING,
    phase="B", prerequisites=(ONTOLOGY,),
    parameters={"convention": "schema_rows", "exempt": ()},
    parameter_choices={"convention": ("schema_rows", "typed_heads")},
    basis="schema_rows: a declared relation named by a schema row whose head type is a declared "
          "term; typed_heads: a relation of the view used on a declared class or typed instance; "
          "declared subclass and type predicates are structure, not properties"))
def mod002(ctx: RuleContext) -> Outcome:
    return _by_schema_rows(ctx) if ctx.params["convention"] == "schema_rows" else _by_typed_heads(ctx)


def _exempt(ctx: RuleContext) -> set[str]:
    """Configured exemptions and the declared subclass and type predicates, by token."""
    structural = (*accepted_predicates(ctx.snapshot, DeclKey.SUBCLASS_PREDICATE),
                  *accepted_predicates(ctx.snapshot, DeclKey.TYPE_PREDICATE))
    return {str(r) for r in ctx.params["exempt"]} | {label(ctx.snapshot, r) for r in structural}


def _by_schema_rows(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    rows = snap.ontology.schema_rows
    if not rows:
        return Outcome.unavailable("ontology_schema_rows")
    relations, entities = relation_resolver(snap), entity_resolver(snap)
    classes = {label_key(n.label) for n in snap.nodes if n.kind is NodeKind.CLASS}
    attributed: dict[str, list[str]] = {}
    for head, rel, _ in rows:
        if entities.declared(head) or label_key(head) in classes:
            for term in relations.targets(rel)[:1]:
                attributed.setdefault(term, []).append(head)
    declared = [r for r in snap.ontology.relations if r not in _exempt(ctx)]
    findings = [_unattributed(ctx, r, "no schema row with a declared head type names it",
                              {"rows": [list(x) for x in rows if relations.targets(x[1])[:1] == [r]]
                               [:_WITNESS]})
                for r in declared if r not in attributed]
    return Outcome(eligible=len(declared), findings=tuple(findings),
                   metrics={"convention": "schema_rows", "schema_rows": len(rows)})


def _by_typed_heads(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    if not snap.has_view(ctx.view):
        return Outcome.unavailable(f"view:{ctx.view.value}")
    if not (snap.capabilities.node_kinds or snap.capabilities.hierarchy):
        return Outcome.unavailable("node_kinds")
    tree = hierarchy(snap, ctx.view)
    heads: dict[str, list[str]] = {}
    for row in snap.view(ctx.view):
        a = snap.assertion_index[row.assertion_id]
        heads.setdefault(a.predicate, []).append(a.head)
    used = [r for r in sorted(heads) if label(snap, r) not in _exempt(ctx)]
    findings = [_unattributed(ctx, label(snap, r), "it is used only on untyped heads",
                              {"heads": sorted({label(snap, h) for h in heads[r]})[:_WITNESS]})
                for r in used if not any(_typed(snap, tree, head) for head in heads[r])]
    return Outcome(eligible=len(used), findings=tuple(findings),
                   metrics={"convention": "typed_heads"})


def _typed(snapshot: AuditSnapshot, h: Hierarchy, node_id: str) -> bool:
    """A declared class, an instance with a declared or asserted type, or a hierarchy member."""
    node = snapshot.node_index.get(node_id)
    return (node is not None and (node.kind is NodeKind.CLASS or bool(node.types))) \
        or node_id in h.types or node_id in h.parents or node_id in h.children()


def _unattributed(ctx: RuleContext, relation: str, why: str, witness: dict[str, object]) -> Finding:
    return ctx.finding(f"relation {relation!r} is attributed to no concept: {why}",
                       affected=(f"relation:{relation}",), locations=(term_ref("relation", relation),)
                       if relation in ctx.snapshot.ontology.relations else (),
                       witness=witness, discriminator="unattributed",
                       suggestion="add a schema row naming the concept the relation describes, "
                                  "or exempt the relation")
