# SPDX-License-Identifier: Apache-2.0
"""META001: problems with the audit schema sidecar.

The sidecar is optional audit metadata, not a store artifact, so its problems
are reported here rather than by ART001 and never fail the store's artifact
contract. A sidecar that does not parse cleanly, breaks its contract or
contradicts itself supplies no metadata; the rules that need it then report
unavailable. This rule also lists declarations that name nothing in the graph,
terms only the sidecar names, and inferred declarations awaiting acceptance.
"""

from __future__ import annotations

from collections.abc import Iterator

from knowlytix.knowledge.kgcheck.audit_schema import SIDECAR
from knowlytix.knowledge.kgcheck.models import (
    Declaration,
    DeclKey,
    DeclSource,
    Diagnostic,
    Finding,
    NodeKind,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin

# keys whose subject must already be a node: these declarations annotate, never add, a term
_REFERENCES = (DeclKey.ROOT, DeclKey.MODULE, DeclKey.CONVENTION_EXCEPTION,
               DeclKey.SUBCLASS_PREDICATE, DeclKey.TYPE_PREDICATE)
_NOTICES = frozenset({"audit_schema_overridden"})


@builtin(RuleSpec(
    "META001", "1", "metadata", "Audit schema problems", Severity.WARNING, phase="B",
    basis="reader diagnostics located in the audit schema sidecar; sidecar declarations checked "
          "against the snapshot's node inventory"))
def meta001(ctx: RuleContext) -> Outcome:
    snap = ctx.snapshot
    read = any(a.name == SIDECAR and a.present for a in snap.artifacts)
    if not read:
        return Outcome(eligible=0)
    sem = snap.semantics
    findings = [*_problems(ctx), *_dangling(ctx), *_sidecar_only(ctx)]
    inferred = sum(d.source is DeclSource.INFERRED for d in sem.declarations)
    if inferred:
        findings.append(ctx.finding(f"{inferred} inferred declarations take no effect until "
                                    "accepted", severity=Severity.INFO, discriminator="inferred",
                                    locations=(sem.declarations[0].raw,) if sem.declarations else (),
                                    witness={"inferred": inferred},
                                    suggestion="review them and set their source to user"))
    return Outcome(eligible=1, findings=tuple(findings),
                   metrics={"usable": sem.version is not None, "declarations": len(sem.declarations),
                            "inferred": inferred})


def _problems(ctx: RuleContext) -> Iterator[Finding]:
    for d in ctx.snapshot.diagnostics:
        if d.location.artifact == SIDECAR:
            yield _problem(ctx, d)


def _problem(ctx: RuleContext, d: Diagnostic) -> Finding:
    notice = d.code in _NOTICES
    message = d.message if notice else f"{d.message}; the audit schema supplies no metadata"
    return ctx.finding(message, severity=Severity.INFO if notice else Severity.WARNING,
                       locations=(d.location,), witness={"code": d.code}, discriminator=d.code,
                       suggestion=None if notice else "correct the audit schema file")


def _dangling(ctx: RuleContext) -> Iterator[Finding]:
    nodes = ctx.snapshot.node_index
    for key in _REFERENCES:
        for d in ctx.snapshot.semantics.of(key):
            if d.subject not in nodes:
                yield _named(ctx, d, f"declares {key.value} for {d.subject}, which names no "
                             "node in the store")


def _sidecar_only(ctx: RuleContext) -> Iterator[Finding]:
    sem = ctx.snapshot.semantics
    for node in ctx.snapshot.nodes:
        if node.declared_by == (SIDECAR,) and node.kind is not NodeKind.CLASS:
            first = next(d for d in sem.declarations if d.subject == node.node_id)
            yield _named(ctx, first, f"{node.node_id} is named only in the audit schema")


def _named(ctx: RuleContext, d: Declaration, message: str) -> Finding:
    return ctx.finding(message, severity=Severity.INFO, locations=(d.raw,),
                       discriminator=f"unmatched:{d.key.value}",
                       suggestion="check the spelling against the stored token")
