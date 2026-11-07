# SPDX-License-Identifier: Apache-2.0
"""Vocabulary rules VOC001-VOC004: label collisions, alias ambiguity, distinctions, membership."""

from __future__ import annotations

from collections.abc import Iterator

from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    Finding,
    GraphView,
    Node,
    NodeKind,
    RawRef,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import (
    NORMALIZATION_VERSION,
    is_numeric_token,
    label_key,
)
from knowlytix.knowledge.kgcheck.registry import (
    ONTOLOGY,
    Outcome,
    RuleContext,
    RuleSpec,
    builtin,
)
from knowlytix.knowledge.kgcheck.rules._common import (
    Resolver,
    artifact,
    entity_resolver,
    relation_resolver,
    rows_by_assertion,
    term_ref,
    triple,
)

_ALL_VIEWS = tuple(GraphView)


# --- VOC001 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "VOC001", "1", "vocabulary", "Label collisions after normalization", Severity.WARNING,
    parameters={"unique_names": False},
    basis=f"labels grouped by kgcheck normalization v{NORMALIZATION_VERSION} within each namespace; "
          "literal nodes excluded"))
def voc001(ctx: RuleContext) -> Outcome:
    groups: dict[tuple[str, str], list[Node]] = {}
    terms = [n for n in ctx.snapshot.nodes if n.kind is not NodeKind.LITERAL]
    for node in terms:
        groups.setdefault((node.namespace, label_key(node.label)), []).append(node)
    severity = Severity.ERROR if ctx.params["unique_names"] else Severity.WARNING
    findings = [_collision(ctx, ns, key, nodes, severity)
                for (ns, key), nodes in sorted(groups.items()) if len(nodes) > 1]
    return Outcome(eligible=len(terms), findings=tuple(findings),
                   metrics={"normalization_version": NORMALIZATION_VERSION,
                            "colliding_groups": len(findings)})


def _collision(ctx: RuleContext, namespace: str, key: str, nodes: list[Node],
               severity: Severity) -> Finding:
    labels = [n.label for n in nodes]
    return ctx.finding(f"{namespace} labels {labels} normalize to one key {key!r}",
                       severity=severity, affected=tuple(n.node_id for n in nodes),
                       witness={"key": key, "labels": {n.label: list(n.declared_by) for n in nodes}},
                       suggestion="merge the spellings or declare them as distinct terms",
                       discriminator=namespace)


# --- VOC002 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "VOC002", "1", "vocabulary", "Alias ambiguity", Severity.ERROR, prerequisites=(ONTOLOGY,),
    basis="index of every declared alias and canonical name by normalized key; an alias that "
          "reaches two terms, or another term's name, has no unique target"))
def voc002(ctx: RuleContext) -> Outcome:
    findings: list[Finding] = []
    count = 0
    for namespace, resolver in (("entity", entity_resolver(ctx.snapshot)),
                                ("relation", relation_resolver(ctx.snapshot))):
        count += sum(len(v) for v in _groups(ctx.snapshot, namespace).values())
        findings += _ambiguous_aliases(ctx, namespace, resolver)
    return Outcome(eligible=count, findings=tuple(findings))


def _groups(snapshot: AuditSnapshot, namespace: str):  # type: ignore[no-untyped-def]
    return snapshot.ontology.entities if namespace == "entity" else snapshot.ontology.relations


def _ambiguous_aliases(ctx: RuleContext, namespace: str, resolver: Resolver) -> Iterator[Finding]:
    reported: set[tuple[str, ...]] = set()
    for term, aliases in _groups(ctx.snapshot, namespace).items():
        for alias in aliases:
            targets = resolver.targets(alias)
            if len(targets) < 2 or (key := (namespace, alias, *targets)) in reported:
                continue
            reported.add(key)
            names = [t for t in targets if t in _named(resolver, alias)]
            yield ctx.finding(
                f"{namespace} alias {alias!r} of {term!r} resolves to {targets}",
                affected=tuple(f"{namespace}:{t}" for t in targets),
                locations=tuple(term_ref(namespace, t) for t in targets),
                witness={"alias": alias, "declared_under": term, "targets": targets,
                         "collides_with_names": names, "producer_resolves_to": targets[0]},
                suggestion="keep the alias under one term only", discriminator=f"{alias}")


def _named(resolver: Resolver, token: str) -> list[str]:
    return [t for k in sorted(resolver.keys(token)) for t in resolver.names.get(k, ())]


# --- VOC003 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "VOC003", "2", "vocabulary", "Collapse of declared distinctions", Severity.ERROR,
    prerequisites=(ONTOLOGY,),
    basis="each declared distinct pair, once however often or in whichever order it is listed, "
          "resolved through names and aliases in every namespace that declares both sides; the "
          "pair collapses when both sides resolve to one term, share an alias or share a key"))
def voc003(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    resolvers = {"entity": entity_resolver(snap), "relation": relation_resolver(snap)}
    pairs: dict[tuple[str, str], list[int]] = {}
    for i, (a, b) in enumerate(snap.ontology.distinct):
        pairs.setdefault((min(a, b), max(a, b)), []).append(i)
    findings = [f for (a, b), rows in sorted(pairs.items())
                for f in _distinct(ctx, rows, a, b, resolvers)]
    return Outcome(eligible=len(pairs), findings=tuple(findings),
                   metrics={"declared_pairs": len(snap.ontology.distinct), "distinct_pairs": len(pairs)})


def _distinct(ctx: RuleContext, rows: list[int], a: str, b: str,
              resolvers: dict[str, Resolver]) -> list[Finding]:
    refs = tuple(RawRef("ontology.json", f"/distinct/{i}") for i in rows)
    both = [(ns, r) for ns, r in resolvers.items() if r.declared(a) and r.declared(b)]
    if not both:
        return [_unchecked(ctx, refs, a, b, resolvers)]
    return [f for ns, r in both if (f := _collapse(ctx, refs, ns, r, a, b))]


def _unchecked(ctx: RuleContext, refs: tuple[RawRef, ...], a: str, b: str,
               resolvers: dict[str, Resolver]) -> Finding:
    missing = [x for x in (a, b) if not any(r.declared(x) for r in resolvers.values())]
    why = f"names undeclared {missing}" if missing else "declares its sides in different namespaces"
    return ctx.finding(f"distinct pair {[a, b]} {why}; the distinction cannot be checked",
                       severity=Severity.WARNING, locations=refs,
                       witness={"pair": [a, b], "undeclared": missing}, discriminator="undeclared")


def _collapse(ctx: RuleContext, refs: tuple[RawRef, ...], namespace: str, resolver: Resolver,
              a: str, b: str) -> Finding | None:
    ta, tb = resolver.targets(a), resolver.targets(b)
    shared = sorted(set(ta) & set(tb))
    common = sorted(_alias_keys(resolver, ta) & _alias_keys(resolver, tb))
    same_key = bool(resolver.keys(a) & resolver.keys(b))
    if not shared and not common and not same_key:
        return None
    via = f"onto {shared}" if shared else f"through shared aliases {common}" if common else "by normalization"
    return ctx.finding(f"declared distinct {namespace}s {a!r} and {b!r} collapse {via}",
                       locations=refs, affected=tuple(f"{namespace}:{t}" for t in sorted(shared or {*ta, *tb})),
                       witness={"pair": [a, b], "targets": {a: ta, b: tb}, "shared_alias_keys": common},
                       suggestion="remove the shared alias or rename one term",
                       discriminator=f"collapse:{namespace}:{a}|{b}")


def _alias_keys(resolver: Resolver, terms: list[str]) -> set[str]:
    return {key for key, owners in resolver.aliases.items() if set(owners) & set(terms)}


# --- VOC004 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "VOC004", "1", "vocabulary", "Vocabulary membership of the selected view", Severity.WARNING,
    prerequisites=(ONTOLOGY,), views=_ALL_VIEWS,
    parameters={"closed_vocabulary": False, "literals_allowed": True},
    basis="each head, relation and object resolved against declared names and aliases; objects "
          "the reader classified as literals are accepted under the literal policy"))
def voc004(ctx: RuleContext) -> Outcome:
    usage = _usage(ctx)
    closed = _closed(ctx)
    severity = Severity.ERROR if closed else Severity.WARNING
    findings = [_undeclared(ctx, ns, token, uses, severity)
                for (ns, token), uses in sorted(usage.undeclared.items())]
    findings += _numeric_heads(ctx, usage) + _trained_literals(ctx)
    return Outcome(eligible=len(usage.tokens), findings=tuple(findings),
                   metrics={"closed_vocabulary": closed, "terms_used": len(usage.tokens),
                            "undeclared_terms": len(usage.undeclared),
                            "literal_objects": len(usage.literals)})


class _Usage:
    def __init__(self) -> None:
        self.tokens: set[tuple[str, str]] = set()
        self.undeclared: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.literals: set[str] = set()
        self.numeric_heads: dict[str, list[str]] = {}


def _usage(ctx: RuleContext) -> _Usage:
    snap, usage = ctx.snapshot, _Usage()
    ent, rel = entity_resolver(snap), relation_resolver(snap)
    allow_literals = bool(ctx.params["literals_allowed"])
    for aid in rows_by_assertion(snap, ctx.view):
        a, (h, r, o) = snap.assertion_index[aid], triple(snap, aid)
        literal = snap.node_index[a.object].kind is NodeKind.LITERAL
        if literal:
            usage.literals.add(o)
        for ns, token, role, ok in (("entity", h, "head", ent.declared(h)),
                                    ("relation", r, "relation", rel.declared(r)),
                                    ("entity", o, "object", ent.declared(o) or (literal and allow_literals))):
            _use(usage, ns, token, role, aid, ok)
        if _numeric(snap, a.head):
            usage.numeric_heads.setdefault(h, []).append(aid)
    return usage


def _use(usage: _Usage, ns: str, token: str, role: str, aid: str, declared: bool) -> None:
    usage.tokens.add((ns, token))
    if not declared:
        usage.undeclared.setdefault((ns, token), []).append((role, aid))


def _numeric(snapshot: AuditSnapshot, node_id: str) -> bool:
    return is_numeric_token(snapshot.node_index[node_id].label)


def _closed(ctx: RuleContext) -> bool:
    return (bool(ctx.params["closed_vocabulary"]) or ctx.view is GraphView.CONFORMANT
            or ctx.snapshot.metadata.get("conformance_policy") == "conformant")


def _undeclared(ctx: RuleContext, ns: str, token: str, uses: list[tuple[str, str]],
                severity: Severity) -> Finding:
    roles = sorted({role for role, _ in uses})
    sample = [list(triple(ctx.snapshot, aid)) for _, aid in uses[:5]]
    return ctx.finding(f"{ns} {token!r} is not declared in ontology.json ({len(uses)} uses as "
                       f"{'/'.join(roles)})", severity=severity, affected=(f"{ns}:{token}",),
                       witness={"uses": len(uses), "roles": roles, "sample_triples": sample},
                       suggestion="declare the term or an alias, or review the extraction",
                       discriminator="undeclared")


def _numeric_heads(ctx: RuleContext, usage: _Usage) -> list[Finding]:
    return [ctx.finding(f"numeric token {head!r} is the subject of {len(aids)} triples",
                        affected=(f"entity:{head}",), discriminator="numeric_head",
                        witness={"sample_triples": [list(triple(ctx.snapshot, a)) for a in aids[:5]]},
                        suggestion="a number is a value, not a subject; review the extraction")
            for head, aids in sorted(usage.numeric_heads.items())]


def _trained_literals(ctx: RuleContext) -> list[Finding]:
    if ctx.view is not GraphView.STORED:
        return []
    vocab = artifact(ctx.snapshot, "adapter.json").get("entity_to_idx")
    tokens = sorted(t for t in (vocab or {}) if (n := ctx.snapshot.node_index.get(f"entity:{t}"))
                    and n.kind is NodeKind.LITERAL)
    if not tokens:
        return []
    by_basis: dict[str, int] = {}
    for t in tokens:
        literal = ctx.snapshot.node_index[f"entity:{t}"].literal
        basis = literal.basis if literal else "unknown"
        by_basis[basis] = by_basis.get(basis, 0) + 1
    kinds = ", ".join(f"{n} {b}" for b, n in sorted(by_basis.items()))
    return [ctx.finding(f"{len(tokens)} of {len(vocab)} trained entity slots hold literals ({kinds})",
                        severity=Severity.INFO, affected=tuple(f"entity:{t}" for t in tokens),
                        locations=(RawRef("adapter.json", "/entity_to_idx"),),
                        witness={"literals": tokens[:50], "entity_slots": len(vocab),
                                 "by_basis": dict(sorted(by_basis.items()))},
                        discriminator="trained_literals",
                        suggestion="literal values are embedded as entities; exact values are "
                                   "served from numeric memory")]
