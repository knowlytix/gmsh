# SPDX-License-Identifier: Apache-2.0
"""Incremental audits: reuse a rule's result when nothing it read has changed.

A full run records, for each rule, exactly which parts of the snapshot it read:
top-level fields, each graph view and each parsed artifact separately, through
a recording proxy that also covers the engine's prerequisite checks. The cache
stores a digest of each part with the result. A later run reuses the result
only when the rule, its version, view, parameters and severity override are the
same and every part it read has the same digest; otherwise the rule runs again.
Because a shipped rule's result is a function of what it reads and its
configuration, a reused result equals the one a full audit would produce.

Three things are outside that argument, so they invalidate or bypass the
cache. A change to kgcheck's own code invalidates every entry (the cache
records a digest of the package source). Rules whose results depend on
something outside the snapshot, such as producer code, a shapes file, a
reasoner or a model, declare ``cacheable=False`` and always run. Rules supplied
by the caller, as extra rules or in a registry of their own, always run, because
their code is not covered by that digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from importlib.resources import files
from pathlib import Path

from knowlytix.knowledge.kgcheck.config import AuditConfig, resolve
from knowlytix.knowledge.kgcheck.frozen import thaw
from knowlytix.knowledge.kgcheck.models import (
    AuditReport,
    AuditSnapshot,
    Finding,
    GraphView,
    JsonObject,
    RawRef,
    RuleResult,
    RuleStatus,
    Severity,
)
from knowlytix.knowledge.kgcheck.registry import Rule, RuleRegistry, builtin_registry

CACHE_VERSION = "kgcheck-audit-cache/1"


# --- what a rule reads ----------------------------------------------------------------
class _ReadMap(Mapping):  # type: ignore[type-arg]
    """A mapping that records which keys are read; iterating it reads every key."""

    def __init__(self, data: Mapping, prefix: str, reads: set[str]) -> None:
        self._data, self._prefix, self._reads = data, prefix, reads

    def __getitem__(self, key: object) -> object:
        self._reads.add(f"{self._prefix}:{_key(key)}")
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        self._reads.add(f"{self._prefix}:{_key(key)}")
        return key in self._data

    def __iter__(self) -> Iterator:  # type: ignore[type-arg]
        self._reads.add(f"{self._prefix}:*")
        return iter(self._data)

    def __len__(self) -> int:
        self._reads.add(f"{self._prefix}:*")
        return len(self._data)


def _key(key: object) -> str:
    return key.value if isinstance(key, Enum) else str(key)


_ALIASES = {"node_index": "nodes", "assertion_index": "assertions"}


class ReadTracker:
    """Stands in for an ``AuditSnapshot`` and records the parts a rule reads."""

    def __init__(self, snapshot: AuditSnapshot) -> None:
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "reads", set())

    def __getattr__(self, name: str) -> object:
        snap: AuditSnapshot = object.__getattribute__(self, "_snapshot")
        reads: set[str] = object.__getattribute__(self, "reads")
        if name in ("parsed", "views"):
            return _ReadMap(getattr(snap, name), name, reads)
        if name in ("view", "has_view"):
            def read_view(view: GraphView) -> object:
                reads.add(f"views:{_key(GraphView(view))}")
                return getattr(snap, name)(view)
            return read_view
        reads.add(_ALIASES.get(name, name))
        return getattr(snap, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a snapshot is read-only")


def part_digest(snapshot: AuditSnapshot, part: str) -> str:
    """Digest of one part of the snapshot, named as ``ReadTracker`` records it."""
    name, _, key = part.partition(":")
    if name in ("parsed", "views"):
        data = getattr(snapshot, name)
        value = (sorted((_key(k), v) for k, v in data.items()) if key == "*" else
                 _lookup(data, key, name))
    else:
        value = getattr(snapshot, name)
    # mapping order is kept: rules read declaration order (the first alias target wins)
    payload = json.dumps(_canonical(value), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


def _lookup(data: Mapping, key: str, name: str) -> object:
    target = GraphView(key) if name == "views" and key in GraphView._value2member_map_ else key
    return ["present", data[target]] if target in data else ["absent"]


def _canonical(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)
                if not f.name.startswith("_")}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return [[_key(k), _canonical(v)] for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN cannot reach a snapshot; guard anyway
        return "NaN"
    return value


# --- the cache ---------------------------------------------------------------------
@cache
def code_digest() -> str:
    """Digest of kgcheck's own source and schemas and of the interpreter and Unicode
    database versions (normalization depends on them); a change invalidates every entry."""
    import sys
    import unicodedata
    root = files("knowlytix.knowledge.kgcheck")
    h = hashlib.sha256(f"{sys.version}|{unicodedata.unidata_version}".encode())
    for path in sorted(_walk(root), key=str):
        h.update(str(path).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _walk(node):  # type: ignore[no-untyped-def]
    for child in node.iterdir():
        if child.is_dir():
            if child.name != "__pycache__":
                yield from _walk(child)
        elif child.name.endswith((".py", ".json")):
            yield child


@dataclass(frozen=True)
class CacheEntry:
    """One rule result with the digests of what the rule read."""

    reads: Mapping[str, str]
    result: JsonObject


@dataclass
class AuditCache:
    """Rule results from earlier audits, keyed by rule configuration. JSON-serializable."""

    code: str = field(default_factory=code_digest)
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return {"cache_version": CACHE_VERSION, "code": self.code,
                "entries": {k: {"reads": dict(sorted(e.reads.items())), "result": e.result}
                            for k, e in sorted(self.entries.items())}}

    @classmethod
    def from_dict(cls, data: object) -> AuditCache:
        """A cache from ``to_dict`` output. One from other code, in another format or
        malformed is empty: a cache only saves work, so a bad one costs a full run."""
        if not isinstance(data, Mapping) or data.get("cache_version") != CACHE_VERSION \
                or data.get("code") != code_digest() or not isinstance(data.get("entries"), Mapping):
            return cls()
        entries = {}
        for key, value in data["entries"].items():
            if _valid_entry(value):
                entries[key] = CacheEntry(dict(value["reads"]), value["result"])
        return cls(code_digest(), entries)

    def save(self, path: str | Path) -> Path:
        """Write atomically, so an interrupted run never leaves a truncated cache."""
        import os
        import tempfile
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, out)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return out

    @classmethod
    def load(cls, path: str | Path) -> AuditCache:
        """The cache at ``path``; an absent, unreadable or malformed file gives an empty cache."""
        source = Path(path)
        if not source.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        except (OSError, ValueError):  # UnicodeDecodeError and JSONDecodeError are ValueErrors
            return cls()


_RESULT_KEYS = ("rule_id", "rule_version", "status", "findings", "eligible", "tested",
                "missing_prerequisites", "error", "runtime_s", "details_truncated", "metrics")


def _valid_entry(value: object) -> bool:
    """An entry ``result_from_dict`` can rebuild; anything else is dropped and recomputed."""
    if not isinstance(value, Mapping) or not isinstance(value.get("reads"), Mapping):
        return False
    result = value.get("result")
    if not isinstance(result, Mapping) or any(k not in result for k in _RESULT_KEYS):
        return False
    try:
        result_from_dict(result)
    except (TypeError, ValueError, KeyError):
        return False
    return all(isinstance(k, str) and isinstance(v, str) for k, v in value["reads"].items())


@dataclass(frozen=True)
class IncrementalStats:
    """Which rules reused a cached result and which ran."""

    reused: tuple[str, ...]
    ran: tuple[str, ...]


def entry_key(rule: Rule, config: AuditConfig, params: Mapping[str, object]) -> str:
    """The configuration a cached result is valid for."""
    severity = config.severity.get(rule.rule_id)
    suppressions = [_canonical(s) for s in config.suppressions if s["rule_id"] == rule.rule_id]
    payload = [rule.rule_id, rule.spec.version, config.view.value, _canonical(params),
               severity.value if severity else None, config.max_findings, suppressions]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def audit_incremental(snapshot: AuditSnapshot, config: AuditConfig | None = None,
                      cache: AuditCache | None = None, *, rules: Iterable[Rule] = (),
                      registry: RuleRegistry | None = None
                      ) -> tuple[AuditReport, AuditCache, IncrementalStats]:
    """``audit`` that reuses cached results whose inputs are unchanged.

    Returns the report, which equals a full audit's, the updated cache and which
    rules were reused. Raises ConfigError for an invalid configuration.
    """
    from knowlytix.knowledge.kgcheck.engine import (
        _disabled,
        _with_extra,
        run_rule,
        snapshot_config,
    )
    config = snapshot_config(snapshot, config)
    cache = cache if cache is not None and cache.code == code_digest() else AuditCache()
    extra = tuple(rules)
    shipped = builtin_registry()  # only shipped code is covered by the code digest
    reg = (registry or shipped).with_rules(extra)
    plan = resolve(_with_extra(config, extra), {rid: r.spec for rid, r in reg.rules().items()})
    results = [_disabled(reg.get(rid)) for rid in plan.disabled]
    reused, ran = [], []
    for rid in plan.run:
        rule, params = reg.get(rid), plan.params[rid]
        cacheable = (rule.spec.cacheable and rid in shipped
                     and shipped.get(rid).check is rule.check and shipped.get(rid).spec == rule.spec)
        key = entry_key(rule, config, params)
        entry = cache.entries.get(key) if cacheable else None
        if entry is not None and all(part_digest(snapshot, p) == d for p, d in entry.reads.items()):
            results.append(result_from_dict(entry.result))
            reused.append(rid)
            continue
        tracker = ReadTracker(snapshot)
        result = run_rule(rule, tracker, config, params)  # type: ignore[arg-type]
        results.append(result)
        ran.append(rid)
        if cacheable and result.status is not RuleStatus.ERROR:
            reads = object.__getattribute__(tracker, "reads")
            cache.entries[key] = CacheEntry({p: part_digest(snapshot, p) for p in sorted(reads)},
                                            result_to_dict(result))
    report = AuditReport.for_snapshot(snapshot, results, profile=config.profile, views=(config.view,))
    return report, cache, IncrementalStats(tuple(reused), tuple(ran))


# --- lossless result serialization -----------------------------------------------------
def result_to_dict(r: RuleResult) -> JsonObject:
    """Every field of a rule result, findings included, as plain JSON."""
    return {"rule_id": r.rule_id, "rule_version": r.rule_version, "status": r.status.value,
            "findings": [_finding(f) for f in r.findings], "eligible": r.eligible,
            "tested": r.tested, "missing_prerequisites": list(r.missing_prerequisites),
            "error": r.error, "runtime_s": r.runtime_s, "details_truncated": r.details_truncated,
            "metrics": thaw(r.metrics)}


def result_from_dict(d: Mapping) -> RuleResult:
    return RuleResult(d["rule_id"], d["rule_version"], RuleStatus(d["status"]),
                      tuple(_finding_from(f) for f in d["findings"]), d["eligible"], d["tested"],
                      tuple(d["missing_prerequisites"]), d["error"], d["runtime_s"],
                      d["details_truncated"], d["metrics"])


def _finding(f: Finding) -> JsonObject:
    return {"rule_id": f.rule_id, "rule_version": f.rule_version, "category": f.category,
            "severity": f.severity.value, "message": f.message, "basis": f.basis,
            "view": f.view.value if f.view else None, "affected": list(f.affected),
            "locations": [[r.artifact, r.pointer] for r in f.locations],
            "witness": thaw(f.witness), "suggestion": f.suggestion,
            "heuristic_score": f.heuristic_score, "discriminator": f.discriminator,
            "suppression": thaw(f.suppression) if f.suppression is not None else None}


def _finding_from(d: Mapping) -> Finding:
    return Finding(d["rule_id"], d["rule_version"], d["category"], Severity(d["severity"]),
                   d["message"], d["basis"], GraphView(d["view"]) if d["view"] else None,
                   tuple(d["affected"]), tuple(RawRef(a, p) for a, p in d["locations"]),
                   d["witness"], d["suggestion"], d["heuristic_score"], d["discriminator"],
                   d.get("suppression"))
