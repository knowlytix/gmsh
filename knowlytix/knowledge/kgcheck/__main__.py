# SPDX-License-Identifier: Apache-2.0
"""Command line: ``python -m knowlytix.knowledge.kgcheck audit <store> [options]``.

``audit-ontology <ontology.json>`` audits an ontology without a store (taxonomy
profile, proposal view), ``audit-rdf <graph> [--shapes <file>]`` audits an RDF
graph (rdf_shacl profile; needs the kgcheck-rdf extra), ``compare <before.json> <after.json>`` compares two reports,
``draft-schema <store> --out <file>`` writes a sidecar
of inferred declarations for review, and ``rules`` lists the shipped rules.

Exit codes: 0 gate pass (or ``--report-only``), 1 gate fail, 2 invalid
invocation or configuration, or a directory with no store artifacts, 3 gate
indeterminate. Reports are written before the exit code is decided.

The audit is read-only, so report destinations are checked before anything is
read or written: an output may not lie inside the store directory (symbolic
links resolved), may not be a hard link to a store file, and the JSON and
Markdown reports may not be one file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from knowlytix.knowledge.kgcheck.config import AuditConfig, ConfigError, load_profile
from knowlytix.knowledge.kgcheck.engine import audit_ontology, audit_rdf, audit_store
from knowlytix.knowledge.kgcheck.gate import GateDecision, GateOutcome, GatePolicy
from knowlytix.knowledge.kgcheck.models import AuditReport, GraphView, Severity
from knowlytix.knowledge.kgcheck.registry import builtin_registry
from knowlytix.knowledge.kgcheck.reporting.json_report import RunInfo

EXIT = {GateOutcome.PASS: 0, GateOutcome.FAIL: 1, GateOutcome.INDETERMINATE: 3}
EXIT_INVALID = 2
_FAIL_ON = [s.value for s in Severity] + ["never"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m knowlytix.knowledge.kgcheck",
                                     description="Read-only audit of an OntoDisc store or ontology.")
    sub = parser.add_subparsers(dest="command", required=True)
    quality = sub.add_parser("quality", help="audit an OntoDisc KG and write a unified quality bundle")
    quality.add_argument("store", type=Path)
    quality.add_argument("--out", type=Path, required=True, help="new report directory outside the store")
    quality.add_argument("--config", type=Path, help="AuditConfig JSON for the store component")
    quality.add_argument("--previous", type=Path, help="previous quality report.json for component comparisons")
    quality.add_argument("--vocabulary", type=Path, help="explicit final ontology JSON")
    run = sub.add_parser("audit", help="audit a store directory")
    run.add_argument("store", type=Path)
    run.add_argument("--cache", type=Path,
                     help="incremental audit: reuse and update the rule results in this file")
    _audit_options(run, profile="ontodisc_default", view="stored")
    onto = sub.add_parser("audit-ontology", help="audit an ontology.json without a store")
    onto.add_argument("store", type=Path, metavar="ontology")
    _audit_options(onto, profile="taxonomy", view="proposal")
    graph = sub.add_parser("audit-rdf", help="audit an RDF graph file (kgcheck-rdf extra)")
    graph.add_argument("store", type=Path, metavar="graph")
    graph.add_argument("--shapes", type=Path, help="SHACL shapes file for CON007")
    graph.add_argument("--format", help="rdflib format name (default: guessed from the suffix)")
    _audit_options(graph, profile="rdf_shacl", view="stored")
    draft = sub.add_parser("draft-schema",
                           help="draft an audit schema of inferred declarations for review")
    draft.add_argument("store", type=Path)
    draft.add_argument("--out", type=Path, required=True, help="write the draft here (new file)")
    draft.add_argument("--view", choices=[v.value for v in GraphView], default="stored")
    compare = sub.add_parser("compare", help="compare two JSON reports")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--json", type=Path, help="write the comparison here")
    rules = sub.add_parser("rules", help="list the shipped rules")
    rules.add_argument("--profile", default="ontodisc_default")
    return parser


def _audit_options(run: argparse.ArgumentParser, *, profile: str, view: str) -> None:
    run.add_argument("--profile", help=f"shipped profile (default {profile})")
    run.add_argument("--view", choices=[v.value for v in GraphView], help=f"graph view (default {view})")
    run.add_argument("--config", type=Path, help="JSON file with AuditConfig fields")
    run.add_argument("--audit-schema", type=Path,
                     help="audit schema sidecar to use instead of the store's audit_schema.json")
    run.add_argument("--rules", help="comma-separated rule ids to run instead of the profile's")
    run.add_argument("--disable", help="comma-separated rule ids to record as disabled")
    run.add_argument("--max-findings", type=int, help="findings kept per rule (default 200)")
    run.add_argument("--json", type=Path, help="write the JSON report here")
    run.add_argument("--markdown", type=Path, help="write the Markdown report here")
    run.add_argument("--no-excerpts", action="store_true", help="omit witnesses from Markdown")
    run.add_argument("--fail-on", choices=_FAIL_ON, default="error",
                     help="lowest severity that fails the gate (default error)")
    run.add_argument("--require", action="append", metavar="RULE",
                     help="rule that must reach a definite result (repeatable; default: profile's)")
    run.add_argument("--report-only", action="store_true", help="exit 0 whatever the gate says")
    run.set_defaults(default_profile=profile, default_view=view)


def _no_constant(token: str) -> object:
    raise ConfigError(f"configuration files must be strict JSON; {token} is not allowed")


def _ids(text: str | None) -> tuple[str, ...] | None:
    return None if text is None else tuple(t.strip() for t in text.split(",") if t.strip())


def _config(args: argparse.Namespace) -> AuditConfig:
    data: dict[str, object] = {"profile": args.default_profile, "view": args.default_view}
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding="utf-8"), parse_constant=_no_constant)
        if not isinstance(loaded, dict):
            raise ConfigError(f"{args.config} must hold a JSON object")
        data.update(loaded)
    overrides = {"profile": args.profile, "view": args.view, "rules": _ids(args.rules),
                 "disabled": _ids(args.disable), "max_findings": args.max_findings,
                 "audit_schema": str(args.audit_schema) if args.audit_schema else None}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return AuditConfig.from_dict(data)


def _policy(args: argparse.Namespace) -> GatePolicy:
    fail_on = None if args.fail_on == "never" else Severity(args.fail_on)
    return GatePolicy(fail_on=fail_on, required_rules=tuple(args.require) if args.require else None)


def _run_info(config: AuditConfig, policy: GatePolicy) -> RunInfo:
    from knowlytix.knowledge import __version__
    gate = {"fail_on": policy.fail_on.value if policy.fail_on else None,
            "required_rules": list(policy.required_rules) if policy.required_rules is not None else None}
    return RunInfo(created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                   tool_version=str(__version__),
                   configuration={"audit": config.to_dict(), "gate": gate})


class OutputPathError(ValueError):
    """A report destination that would overwrite the store or the other report."""


def check_outputs(store: Path, outputs: Sequence[Path], inputs: Sequence[Path] = ()) -> None:
    """Refuse report destinations that collide with the store, an input file or each other."""
    root = store.resolve()
    resolved = [out.resolve() for out in outputs]
    for out, target in zip(outputs, resolved, strict=True):
        if target == root or target.is_relative_to(root):
            raise OutputPathError(f"{out} is inside the audited store {store}")
        if any(target == i.resolve() or (target.exists() and i.exists() and target.samefile(i))
               for i in inputs):
            raise OutputPathError(f"{out} would overwrite an input of the audit")
        if target.exists() and _links_into(target, root):
            raise OutputPathError(f"{out} is a hard link to a file of the audited store")
    if len(set(resolved)) < len(resolved) or _same_existing(resolved):
        raise OutputPathError("the JSON and Markdown reports must be different files")


def _links_into(target: Path, root: Path) -> bool:
    if not root.is_dir() or target.stat().st_nlink < 2:
        return False
    return any(p.is_file() and p.samefile(target) for p in root.rglob("*"))


def _same_existing(paths: list[Path]) -> bool:
    existing = [p for p in paths if p.exists()]
    return any(a.samefile(b) for i, a in enumerate(existing) for b in existing[i + 1:])


def _write(args: argparse.Namespace, report: AuditReport, gate: GateDecision, run: RunInfo) -> list[Path]:
    written = []
    if args.json is not None:
        written.append(report.write_json(args.json, run, gate))
    if args.markdown is not None:
        written.append(report.write_markdown(args.markdown, gate, excerpts=not args.no_excerpts))
    return written


def _summary(store: Path, report: AuditReport, gate: GateDecision, written: list[Path]) -> str:
    statuses = ", ".join(f"{s.value} {len(ids)}" for s, ids in report.by_status().items())
    totals = report.finding_counts().by_severity
    findings = ", ".join(f"{s.value} {totals[s.value]}" for s in Severity if s.value in totals) or "none"
    lines = [f"kgcheck: {store} snapshot {report.snapshot_id} profile {report.profile} "
             f"view {', '.join(v.value for v in report.views)}",
             f"rules: {len(report.results)} ({statuses})", f"findings: {findings}",
             f"execution: {'complete' if report.is_complete() else 'incomplete'}",
             f"gate: {gate.outcome.value}" + _gate_detail(gate)]
    return "\n".join(lines + [f"wrote {p}" for p in written])


def _gate_detail(gate: GateDecision) -> str:
    parts = [f"{len(gate.blocking_findings)} blocking finding(s)"] if gate.blocking_findings else []
    if gate.unmet_required:
        parts.append("required: " + ", ".join(f"{r} {s}" for r, s in gate.unmet_required.items()))
    if gate.snapshot_findings:
        parts.append("store changed during the read")
    return f" ({'; '.join(parts)})" if parts else ""


def _audit(args: argparse.Namespace) -> int:
    ontology, graph = args.command == "audit-ontology", args.command == "audit-rdf"
    config, policy = _config(args), _policy(args)
    sidecar = Path(config.audit_schema) if config.audit_schema else None
    shapes, cache = getattr(args, "shapes", None), getattr(args, "cache", None)
    if cache is not None and cache.exists() and not cache.is_file():
        raise OutputPathError(f"--cache {cache} is not a file")
    check_outputs(_protected_root(args.store) if ontology else args.store,
                  [p for p in (args.json, args.markdown, cache) if p is not None],
                  [p for p in (sidecar, args.config, shapes,
                               args.store if ontology or graph else None) if p is not None])
    if graph:
        report = audit_rdf(args.store, config, format=args.format, shapes=shapes)
    elif cache is not None:
        report = _audit_cached(args.store, config, cache)
    else:
        report = (audit_ontology(args.store, None, config) if ontology
                  else audit_store(args.store, config))
    if report.profile != config.profile:
        config = replace(config, profile=report.profile)
    gate = report.evaluate_gate(policy)
    written = _write(args, report, gate, _run_info(config, policy))
    print(_summary(args.store, report, gate, written))
    if not (ontology or graph) and not any(a.present for a in report.artifacts
                                           if a.name != "audit_schema.json"):
        print(f"kgcheck: error: {args.store} holds none of the store artifacts", file=sys.stderr)
        return EXIT_INVALID
    return 0 if args.report_only else EXIT[gate.outcome]


def _audit_cached(store: Path, config: AuditConfig, path: Path) -> AuditReport:
    from knowlytix.knowledge.kgcheck.adapters import read_store
    from knowlytix.knowledge.kgcheck.incremental import AuditCache, audit_incremental
    sidecar = Path(config.audit_schema) if config.audit_schema else None
    report, cache, stats = audit_incremental(read_store(store, sidecar), config, AuditCache.load(path))
    cache.save(path)
    print(f"kgcheck: incremental: {len(stats.reused)} rule results reused, {len(stats.ran)} run")
    return report


def _protected_root(ontology: Path) -> Path:
    """The store directory when the ontology lies in one, else the ontology file itself."""
    from knowlytix.knowledge.kgcheck.contract import artifact_specs
    folder = ontology.resolve().parent
    others = [s.filename for s in artifact_specs() if s.filename != "ontology.json"]
    return folder if any((folder / name).exists() for name in others) else ontology


def _draft(args: argparse.Namespace) -> int:
    check_outputs(args.store, [args.out])
    if args.out.exists():
        raise OutputPathError(f"{args.out} exists; the draft is written only to a new file")
    from knowlytix.knowledge.kgcheck.adapters import read_store
    from knowlytix.knowledge.kgcheck.draft import draft_audit_schema
    snapshot = read_store(args.store)
    if not any(a.present for a in snapshot.artifacts if a.name != "audit_schema.json"):
        print(f"kgcheck: error: {args.store} holds none of the store artifacts", file=sys.stderr)
        return EXIT_INVALID
    doc = draft_audit_schema(snapshot, GraphView(args.view))
    try:  # encode before opening, so a failure leaves no empty file behind
        data = (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate in a stored label
        data = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
    with args.out.open("xb") as out:
        out.write(data)
    print(f"kgcheck: wrote a draft of {_count_entries(doc)} inferred entries to {args.out}; "
          "review each entry and set its source to user to accept it")
    return 0


def _count_entries(doc: dict[str, object]) -> int:
    return sum(len(v) for k, v in doc.items() if isinstance(v, dict))


def _compare(args: argparse.Namespace) -> int:
    from knowlytix.knowledge.kgcheck.reporting.compare import compare_reports
    before, after = (json.loads(p.read_text(encoding="utf-8")) for p in (args.before, args.after))
    result = compare_reports(before, after).to_dict()
    if args.json is not None:
        if args.json.resolve() in (args.before.resolve(), args.after.resolve()):
            raise OutputPathError("the comparison may not overwrite a compared report")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = ", ".join(f"{k} {v}" for k, v in result["counts"].items() if v)
    print(f"kgcheck: {result['before']} -> {result['after']}: {counts or 'no findings'}"
          + ("" if result["comparable"] else f" (not comparable: {'; '.join(result['reasons'])})"))
    return 0


def _list_rules(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    for rule_id, rule in builtin_registry().rules().items():
        spec = rule.spec
        marker = "required" if rule_id in profile.required_rules else (
            "profile" if rule_id in profile.rules else "")
        print(f"{rule_id}  v{spec.version}  {spec.category:<10} {spec.default_severity.value:<8} "
              f"{marker:<8} {spec.title}")
    return 0


def _quality(args):
    from .quality import _read_metadata, audit_quality
    config = AuditConfig.from_dict(_read_metadata(args.config)[0]) if args.config else None
    vocabulary = _read_metadata(args.vocabulary)[0] if args.vocabulary else None
    check_outputs(args.store, [args.out], [p for p in (args.config, args.vocabulary) if p])
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f"Use a new quality report destination: {args.out}")
    report = audit_quality(args.store, vocabulary=vocabulary, config=config)
    report.write_bundle(args.out, store=args.store, previous=args.previous)
    print(f"kgcheck quality: {report.outcome}; report: {args.out / 'report.md'}")
    return {"pass": 0, "fail": 1, "indeterminate": 3}[report.outcome]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        commands = {"quality": _quality, "audit": _audit, "audit-ontology": _audit, "audit-rdf": _audit,
                    "draft-schema": _draft, "compare": _compare,
                    "rules": _list_rules}
        return commands[args.command](args)
    except (ConfigError, OutputPathError, FileNotFoundError, ValueError, OSError,
            ImportError) as exc:  # ImportError: an optional backend is not installed
        print(f"kgcheck: error: {exc}", file=sys.stderr)  # JSON and Unicode errors are ValueErrors
        return EXIT_INVALID
    except Exception as exc:  # noqa: BLE001  # a crash must not read as a gate failure (exit 1)
        print(f"kgcheck: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
