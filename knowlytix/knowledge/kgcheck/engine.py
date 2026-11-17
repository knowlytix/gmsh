# SPDX-License-Identifier: Apache-2.0
"""Run rules on a snapshot and collect their results into a report.

The engine owns status semantics. It records a disabled rule as ``disabled``,
a rule whose prerequisites the snapshot lacks or that does not support the
selected view as ``unavailable``, a rule that
raised as ``error``, and otherwise derives the status from the rule's outcome:
``partial`` when it stopped early, ``findings`` when it found something,
``not_applicable`` when it had no targets and ``passed`` only when it examined
at least one target and found nothing. A failing rule never stops the others.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from knowlytix.knowledge.kgcheck.config import AuditConfig, resolve
from knowlytix.knowledge.kgcheck.frozen import FrozenJson
from knowlytix.knowledge.kgcheck.models import (
    AuditReport,
    AuditSnapshot,
    Capabilities,
    Finding,
    GraphView,
    RuleResult,
    RuleResultError,
    RuleStatus,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import (
    ONTOLOGY,
    VIEW,
    Outcome,
    Rule,
    RuleContext,
    RuleRegistry,
    builtin_registry,
)

TAXONOMY_PROFILE = "taxonomy"
TRIPLES_PROFILE = "triples"
RDF_PROFILE = "rdf_shacl"
SEVERITY_RANK = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2, Severity.SUGGESTION: 3}
_CAPABILITIES = frozenset(f.name for f in dataclasses.fields(Capabilities)
                          if f.type in ("bool", bool))


def audit(snapshot: AuditSnapshot, config: AuditConfig | None = None, *,
          rules: Iterable[Rule] = (), registry: RuleRegistry | None = None) -> AuditReport:
    """Audit ``snapshot`` with the configured profile plus any extra ``rules``.

    Extra rules always run (unless disabled) in addition to the profile's.
    Raises ConfigError for an invalid configuration, including an extra rule
    whose id is already registered; rule failures are reported.
    """
    config = snapshot_config(snapshot, config)
    extra = tuple(rules)
    reg = (registry or builtin_registry()).with_rules(extra)
    specs = {rid: r.spec for rid, r in reg.rules().items()}
    plan = resolve(_with_extra(config, extra), specs)
    results = [_disabled(reg.get(rid)) for rid in plan.disabled]
    results += [run_rule(reg.get(rid), snapshot, config, plan.params[rid]) for rid in plan.run]
    return AuditReport.for_snapshot(snapshot, results, profile=config.profile, views=(config.view,))


def snapshot_config(snapshot, config=None):
    """Select the native profile only for the generic default; preserve explicit rules."""
    config = config or AuditConfig()
    if (snapshot.capabilities.adapter == 'source_assertions'
            and config.profile == 'ontodisc_default' and config.rules is None):
        return dataclasses.replace(config, profile='source_assertions')
    return config


def audit_store(store_dir: str | Path, config: AuditConfig | None = None, *,
                rules: Iterable[Rule] = (), registry: RuleRegistry | None = None) -> AuditReport:
    """Read the OntoDisc store at ``store_dir`` (JSON and Markdown only) and audit it.

    Raises FileNotFoundError when ``store_dir`` is not a directory and ConfigError
    for an invalid configuration; every problem inside the store is reported.
    ``config.audit_schema`` names a sidecar to use instead of the store's own.
    """
    from knowlytix.knowledge.kgcheck.adapters import read_store
    sidecar = config.audit_schema if config is not None else None
    return audit(read_store(store_dir, sidecar), config, rules=rules, registry=registry)


def audit_ontology(ontology: str | Path | Mapping[str, object],
                   metadata: str | Path | Mapping[str, object] | None = None,
                   config: AuditConfig | None = None, *, rules: Iterable[Rule] = (),
                   registry: RuleRegistry | None = None) -> AuditReport:
    """Audit an ontology without a store: ``ontology.json`` (a path or its parsed JSON).

    ``metadata`` is an audit schema sidecar, as a path or parsed JSON (default:
    ``config.audit_schema``). The schema
    rows form the ``proposal`` view, and the default configuration is the
    ``taxonomy`` profile on that view. Raises FileNotFoundError when a path names
    no file and ConfigError for an invalid configuration.
    """
    from knowlytix.knowledge.kgcheck.adapters.ontology_only import read_ontology
    if metadata is None and config is not None:
        metadata = config.audit_schema
    config = config or AuditConfig(profile=TAXONOMY_PROFILE, view=GraphView.PROPOSAL)
    return audit(read_ontology(ontology, metadata), config, rules=rules, registry=registry)


def audit_triples(triples: Iterable[Sequence[str]], ontology: Mapping[str, object] | None = None,
                  metadata: Mapping[str, object] | None = None, config: AuditConfig | None = None,
                  *, rules: Iterable[Rule] = (), registry: RuleRegistry | None = None) -> AuditReport:
    """Audit triples held in memory, with an optional ontology and audit schema sidecar.

    ``triples`` are (head, relation, object) token triples, the ``stored`` view;
    ``ontology`` has the form of ``ontology.json`` and ``metadata`` of the
    sidecar. The default configuration is the ``triples`` profile. Without an
    ontology the vocabulary rules are unavailable, so unused ontology terms and
    vocabulary conformance are not checked. Raises ValueError for a row that is not
    three nonblank strings.
    """
    import json
    import tempfile

    from knowlytix.knowledge.kgcheck.adapters import read_store
    rows = [list(t) for t in triples]
    bad = [i for i, r in enumerate(rows) if len(r) != 3 or not all(isinstance(x, str) and x for x in r)]
    if bad:  # in-memory input is the caller's: reject it rather than audit part of it
        raise ValueError(f"triples must be three nonblank strings; rows {bad[:10]} are not")
    with tempfile.TemporaryDirectory(prefix="kgcheck-triples-") as tmp:
        root = Path(tmp)
        files = {"triples.json": rows, "ontology.json": ontology,
                 "audit_schema.json": metadata or None}  # an empty sidecar is no sidecar
        for name, value in files.items():
            if value is not None:
                (root / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        snapshot = read_store(root, partial=True)
    return audit(snapshot, config or AuditConfig(profile=TRIPLES_PROFILE), rules=rules,
                 registry=registry)


def audit_rdf(path: str | Path, config: AuditConfig | None = None, *, format: str | None = None,
              shapes: str | Path | None = None, rules: Iterable[Rule] = (),
              registry: RuleRegistry | None = None) -> AuditReport:
    """Audit the RDF file at ``path`` (needs ``rdflib``; CON007 also ``pyshacl``).

    The default configuration is the ``rdf_shacl`` profile. ``shapes`` names a
    SHACL shapes file for CON007; ``config.audit_schema`` a sidecar. Raises
    FileNotFoundError when a path names no file, ImportError without rdflib and
    ConfigError for an invalid configuration.
    """
    from knowlytix.knowledge.kgcheck.adapters.rdf_input import read_rdf
    config = config or AuditConfig(profile=RDF_PROFILE)
    if shapes is not None:
        con007 = {**config.params.get("CON007", {}), "shapes": str(shapes)}
        config = dataclasses.replace(config, params={**config.params, "CON007": con007})
    snapshot = read_rdf(path, format=format, audit_schema=config.audit_schema)
    return audit(snapshot, config, rules=rules, registry=registry)


def _with_extra(config: AuditConfig, extra: tuple[Rule, ...]) -> AuditConfig:
    if not extra:
        return config
    from knowlytix.knowledge.kgcheck.config import load_profile
    base = config.rules if config.rules is not None else load_profile(config.profile).rules
    return dataclasses.replace(config, rules=(*base, *(r.rule_id for r in extra)))


def _disabled(rule: Rule) -> RuleResult:
    return RuleResult(rule.rule_id, rule.spec.version, RuleStatus.DISABLED)


def missing_prerequisites(snapshot: AuditSnapshot, view: GraphView,
                          prerequisites: Iterable[str]) -> tuple[str, ...]:
    """The prerequisites ``snapshot`` does not meet, in declaration order."""
    return tuple(p for p in prerequisites if not _met(snapshot, view, p))


def _met(snapshot: AuditSnapshot, view: GraphView, prerequisite: str) -> bool:
    if prerequisite == VIEW:
        return snapshot.has_view(view)
    if prerequisite == ONTOLOGY:
        return isinstance(snapshot.parsed.get("ontology.json"), Mapping)
    if prerequisite.startswith("artifact:"):
        return prerequisite.removeprefix("artifact:") in snapshot.parsed
    if prerequisite in _CAPABILITIES:
        return bool(getattr(snapshot.capabilities, prerequisite))
    raise ValueError(f"unknown prerequisite {prerequisite!r}")


def run_rule(rule: Rule, snapshot: AuditSnapshot, config: AuditConfig,
             params: Mapping[str, FrozenJson]) -> RuleResult:
    """One rule execution, isolated: any failure becomes an ``error`` result."""
    spec, start = rule.spec, time.perf_counter()
    try:
        result = _execute(rule, snapshot, config, params)
    except Exception as exc:  # noqa: BLE001  # a rule failure is reported, not raised
        result = RuleResult(spec.rule_id, spec.version, RuleStatus.ERROR,
                            error=f"{type(exc).__name__}: {exc}"[:500])
    return dataclasses.replace(result, runtime_s=round(time.perf_counter() - start, 6))


def _execute(rule: Rule, snapshot: AuditSnapshot, config: AuditConfig,
             params: Mapping[str, FrozenJson]) -> RuleResult:
    spec = rule.spec
    if spec.views is not None and config.view not in spec.views:
        return RuleResult(spec.rule_id, spec.version, RuleStatus.UNAVAILABLE,
                          missing_prerequisites=(f"view:{config.view.value} unsupported",),
                          metrics={"unsupported_view": config.view.value})
    missing = missing_prerequisites(snapshot, config.view,
                                    (*spec.prerequisites, *((VIEW,) if spec.views else ())))
    if missing:
        return RuleResult(spec.rule_id, spec.version, RuleStatus.UNAVAILABLE,
                          missing_prerequisites=missing)
    outcome = rule.check(RuleContext(snapshot, config.view, spec, params))
    if not isinstance(outcome, Outcome):
        raise TypeError(f"check returned {type(outcome).__name__}, not Outcome")
    return _result(rule, outcome, config)


def _result(rule: Rule, outcome: Outcome, config: AuditConfig) -> RuleResult:
    spec = rule.spec
    if outcome.missing:
        return RuleResult(spec.rule_id, spec.version, RuleStatus.UNAVAILABLE,
                          missing_prerequisites=outcome.missing, metrics=outcome.metrics)
    findings = _override(outcome.findings, config.severity.get(spec.rule_id))
    _check_unique(findings)
    findings, stale = _suppress(spec, findings, config)
    kept = tuple(sorted(findings, key=_order)[:config.max_findings])
    metrics = {**outcome.metrics, "findings_total": len(findings),
               "findings_by_severity": _by_severity(findings),
               "findings_by_category": dict(sorted(Counter(f.category for f in findings).items()))}
    if any(s["rule_id"] == spec.rule_id for s in config.suppressions):
        metrics["suppressed"] = sum(f.suppression is not None for f in findings)
        metrics["stale_suppressions"] = stale
    return RuleResult(spec.rule_id, spec.version, _status(outcome), kept,
                      eligible=outcome.eligible, tested=outcome.examined,
                      error=outcome.partial, details_truncated=len(kept) < len(findings),
                      metrics=metrics)


def _suppress(spec: object, findings: tuple[Finding, ...], config: AuditConfig
              ) -> tuple[tuple[Finding, ...], list[str]]:
    """Mark findings a configured suppression matches; a suppression reviewed under
    another rule version is stale and matches nothing, so the finding is reviewed again."""
    mine = [s for s in config.suppressions if s["rule_id"] == getattr(spec, "rule_id")]
    current = [s for s in mine if s["rule_version"] == getattr(spec, "version")]
    stale = sorted(str(s.get("finding_id") or "|".join(s["affected"])) for s in mine
                   if s not in current)
    out = []
    for f in findings:
        match = next((s for s in current if _matches(s, f)), None)
        out.append(dataclasses.replace(f, suppression={"reason": match["reason"],
                                                       "rule_version": match["rule_version"]})
                   if match else f)
    return tuple(out), stale


def _matches(suppression: Mapping[str, object], f: Finding) -> bool:
    view = suppression.get("view")
    if view is not None and (f.view is None or f.view.value != view):
        return False
    if "finding_id" in suppression:
        return suppression["finding_id"] == f.finding_id
    if "discriminator" in suppression and suppression["discriminator"] != f.discriminator:
        return False
    return sorted(suppression["affected"]) == sorted(f.affected)  # type: ignore[arg-type]


def _status(outcome: Outcome) -> RuleStatus:
    if outcome.partial or outcome.examined < outcome.eligible:
        return RuleStatus.PARTIAL
    if outcome.findings:
        return RuleStatus.FINDINGS
    return RuleStatus.NOT_APPLICABLE if outcome.eligible == 0 else RuleStatus.PASSED


def _override(findings: tuple[Finding, ...], severity: Severity | None) -> tuple[Finding, ...]:
    if severity is None:
        return findings
    return tuple(dataclasses.replace(f, severity=severity) for f in findings)


def _check_unique(findings: tuple[Finding, ...]) -> None:
    ids = [f.finding_id for f in findings]
    if len(ids) != len(set(ids)):
        raise RuleResultError("the rule produced findings with duplicate ids; "
                              "distinct issues on one target need a discriminator")


def _order(f: Finding) -> tuple[bool, int, str]:
    """Unsuppressed first, then most severe: a truncated list keeps what blocks."""
    return f.suppression is not None, SEVERITY_RANK[f.severity], f.finding_id


def _by_severity(findings: tuple[Finding, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
    return dict(sorted(counts.items()))
