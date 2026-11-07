# SPDX-License-Identifier: Apache-2.0
"""Lexical rules VOC005-VOC006: near-duplicate labels and naming conventions.

VOC005 lists label pairs close enough to be the same term spelled differently.
The search is exact under its stated thresholds: candidate pairs come from a
token index (pairs sharing a token) and a length window (pairs whose lengths
differ by at most the edit bound), and every candidate is scored. A work limit
stops the search and marks it partial, never silently incomplete. Pairs that
already resolve to one declared term through its aliases, and declared distinct
pairs, are not suggested. The
thresholds are configurable baselines, not validated cutoffs, and a pair is a
review candidate, never a merge.

VOC006 checks labels against the case and separator conventions the audit
schema sidecar declares per namespace. Singular and plural need a language
analyzer, which kgcheck does not ship, so a number convention is reported as
not checked.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from knowlytix.knowledge.kgcheck.models import (
    AuditSnapshot,
    DeclKey,
    Finding,
    Node,
    NodeKind,
    Severity,
)
from knowlytix.knowledge.kgcheck.normalize import (
    NORMALIZATION_VERSION,
    is_numeric_token,
    label_key,
)
from knowlytix.knowledge.kgcheck.registry import Outcome, RuleContext, RuleSpec, builtin
from knowlytix.knowledge.kgcheck.rules._common import (
    Resolver,
    entity_resolver,
    relation_resolver,
)

_NAMESPACES = ("entity", "relation")
_DIGITS = re.compile(r"\d+")
_SEPARATORS = ("_", "-", " ")


def _terms(snapshot: AuditSnapshot, namespace: str) -> list[Node]:
    """Non-literal nodes of one namespace; numeric tokens are values, not names."""
    return [n for n in snapshot.nodes if n.namespace == namespace
            and n.kind is not NodeKind.LITERAL and not is_numeric_token(n.label)]


# --- VOC005 -------------------------------------------------------------------------
@builtin(RuleSpec(
    "VOC005", "2", "vocabulary", "Near-duplicate labels", Severity.SUGGESTION, phase="B",
    parameters={"max_edit_distance": 2, "max_normalized_distance": 0.2,
                "min_token_overlap": 0.75, "ignore_digit_differences": True,
                "max_comparisons": 2_000_000},
    parameter_bounds={"max_edit_distance": (0, None), "max_normalized_distance": (0.0, 1.0),
                      "min_token_overlap": (0.0, 1.0), "max_comparisons": (1, None)},
    basis=f"normalized labels (kgcheck normalization v{NORMALIZATION_VERSION}) compared within each "
          "namespace by bounded edit distance, edit distance over the longer label, and token "
          "Jaccard overlap; candidates from a token index and a length window"))
def voc005(ctx: RuleContext) -> Outcome:
    search = _Search(ctx)
    for namespace in _NAMESPACES:
        search.run(namespace, _terms(ctx.snapshot, namespace))
    partial = (f"stopped after {search.comparisons} comparisons (max_comparisons)"
               if search.stopped else None)
    return Outcome(eligible=search.keys, tested=search.keys if not search.stopped else search.done,
                   findings=tuple(search.findings), partial=partial,
                   metrics={"normalization_version": NORMALIZATION_VERSION,
                            "comparisons": search.comparisons, "candidate_pairs": len(search.findings),
                            "skipped_distinct_pairs": search.skipped_distinct,
                            "skipped_digit_pairs": search.skipped_digits,
                            "skipped_resolved_pairs": search.skipped_resolved})


@dataclass
class _Pair:
    edit: int | None
    normalized: float | None
    overlap: float


@dataclass
class _Search:
    ctx: RuleContext
    keys: int = 0
    done: int = 0
    comparisons: int = 0
    stopped: bool = False
    skipped_distinct: int = 0
    skipped_digits: int = 0
    skipped_resolved: int = 0
    findings: list[Finding] = field(default_factory=list)

    def run(self, namespace: str, nodes: list[Node]) -> None:
        by_key: dict[str, list[Node]] = {}
        for node in nodes:
            by_key.setdefault(label_key(node.label), []).append(node)
        self.keys += len(by_key)
        if self.stopped:
            return
        distinct = _distinct_keys(self.ctx.snapshot)
        resolver = (relation_resolver if namespace == "relation" else entity_resolver)(self.ctx.snapshot)
        for a, b in _candidates(sorted(by_key), int(self.ctx.params["max_edit_distance"])):
            if self.stopped or self._limit():
                return
            if frozenset((a, b)) in distinct:
                self.skipped_distinct += 1
            elif _same_term(resolver, by_key[a], by_key[b]):
                self.skipped_resolved += 1
            elif self.ctx.params["ignore_digit_differences"] and _digits(a) != _digits(b):
                self.skipped_digits += 1
            elif (pair := self._score(a, b)) is not None:
                self.findings.append(_similar(self.ctx, namespace, by_key[a], by_key[b], pair))
        self.done += len(by_key)

    def _limit(self) -> bool:
        self.comparisons += 1
        self.stopped = self.comparisons > int(self.ctx.params["max_comparisons"])
        return self.stopped

    def _score(self, a: str, b: str) -> _Pair | None:
        p = self.ctx.params
        edit = bounded_edit_distance(a, b, int(p["max_edit_distance"]))
        normalized = None if edit is None else round(edit / max(len(a), len(b)), 4)
        overlap = token_overlap(a, b)
        close = normalized is not None and normalized <= float(p["max_normalized_distance"])
        if close or overlap >= float(p["min_token_overlap"]):
            return _Pair(edit, normalized, overlap)
        return None


def _candidates(keys: list[str], bound: int) -> Iterator[tuple[str, str]]:
    """Each unordered pair that shares a token or lies within the length window, once."""
    seen: set[tuple[str, str]] = set()
    by_token: dict[str, list[str]] = {}
    for key in keys:
        for token in set(key.split()):
            by_token.setdefault(token, []).append(key)
    by_length = sorted(keys, key=lambda k: (len(k), k))
    lengths = [len(k) for k in by_length]
    for key in keys:
        window = by_length[bisect_left(lengths, len(key) - bound):bisect_right(lengths, len(key) + bound)]
        sharing = (k for t in set(key.split()) for k in by_token[t])
        for other in (*window, *sharing):
            pair = (key, other) if key < other else (other, key)
            if key != other and pair not in seen:
                seen.add(pair)
                yield pair


def bounded_edit_distance(a: str, b: str, bound: int) -> int | None:
    """Levenshtein distance of ``a`` and ``b`` when at most ``bound``, else None."""
    if abs(len(a) - len(b)) > bound:
        return None
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > bound:
            return None
        previous = current
    return previous[-1] if previous[-1] <= bound else None


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the two labels' token sets."""
    ta, tb = set(a.split()), set(b.split())
    return round(len(ta & tb) / len(ta | tb), 4) if ta | tb else 0.0


def _same_term(resolver: Resolver, a: list[Node], b: list[Node]) -> bool:
    """Both spellings already resolve to one declared term (a name or an alias of it)."""
    first = {t for n in a for t in resolver.targets(n.label)[:1]}
    return bool(first) and first == {t for n in b for t in resolver.targets(n.label)[:1]}


def _digits(key: str) -> list[str]:
    return _DIGITS.findall(key)


def _distinct_keys(snapshot: AuditSnapshot) -> set[frozenset[str]]:
    return {frozenset((label_key(a), label_key(b))) for a, b in snapshot.ontology.distinct}


def _similar(ctx: RuleContext, namespace: str, a: list[Node], b: list[Node], pair: _Pair) -> Finding:
    la, lb = a[0].label, b[0].label
    score = max(1 - pair.normalized if pair.normalized is not None else 0.0, pair.overlap)
    matched = [m for m, ok in (("edit_distance", pair.normalized is not None
                                and pair.normalized <= float(ctx.params["max_normalized_distance"])),
                               ("token_overlap", pair.overlap >= float(ctx.params["min_token_overlap"])))
               if ok]
    finding = ctx.finding(
        f"{namespace} labels {la!r} and {lb!r} are similar (edit distance {pair.edit}, "
        f"token overlap {pair.overlap})",
        affected=tuple(n.node_id for n in (*a, *b)),
        witness={"labels": [n.label for n in (*a, *b)], "keys": [label_key(la), label_key(lb)],
                 "edit_distance": pair.edit, "normalized_distance": pair.normalized,
                 "token_overlap": pair.overlap, "matched_by": matched},
        suggestion="review: declare one as an alias of the other, or declare the pair distinct",
        discriminator=namespace)
    return replace(finding, heuristic_score=round(score, 4))  # uncalibrated similarity


# --- VOC006 -------------------------------------------------------------------------
_CASE = {"lower": str.lower, "upper": str.upper}


@builtin(RuleSpec(
    "VOC006", "1", "vocabulary", "Naming conventions", Severity.SUGGESTION, phase="B",
    prerequisites=("naming_conventions",),
    basis="each non-literal label checked against the case and separator conventions the audit "
          "schema declares for its namespace; declared exceptions are skipped"))
def voc006(ctx: RuleContext) -> Outcome:
    sem = ctx.snapshot.semantics.accepted()
    declared = {ns: {k: sem.values(k, ns)[0] for k in (DeclKey.CASE, DeclKey.SEPARATOR, DeclKey.NUMBER)
                     if sem.values(k, ns)} for ns in _NAMESPACES}
    checkable = {ns: {k: v for k, v in rules.items() if k is not DeclKey.NUMBER}
                 for ns, rules in declared.items()}
    if not any(checkable.values()):
        return Outcome.unavailable("language_analyzer")
    exceptions = set(sem.subjects(DeclKey.CONVENTION_EXCEPTION))
    nodes = [n for ns, rules in checkable.items() if rules
             for n in _terms(ctx.snapshot, ns) if n.node_id not in exceptions]
    findings = [f for n in nodes if (f := _naming(ctx, n, checkable[n.namespace]))]
    unchecked = sorted(ns for ns, rules in declared.items() if DeclKey.NUMBER in rules)
    return Outcome(eligible=len(nodes), findings=tuple(findings),
                   metrics={"exceptions": len(exceptions),
                            "number_not_checked": {"namespaces": unchecked,
                                                   "reason": "no language analyzer"}})


def _naming(ctx: RuleContext, node: Node, rules: dict[DeclKey, str]) -> Finding | None:
    broken = {key.value: expected for key, expected in rules.items()
              if not _follows(node.label, key, expected)}
    if not broken:
        return None
    return ctx.finding(f"{node.namespace} label {node.label!r} does not follow the declared "
                       f"{', '.join(f'{k} {v!r}' for k, v in broken.items())}",
                       affected=(node.node_id,), witness={"label": node.label, "expected": broken},
                       suggestion="rename the term, or list it under the convention's exceptions",
                       discriminator="naming")


def _follows(label: str, key: DeclKey, expected: str) -> bool:
    if key is DeclKey.SEPARATOR:
        return not any(s in label for s in _SEPARATORS if s != expected)
    if expected == "title":
        return all(w[:1] == w[:1].upper() for w in re.split(r"[_\- ]+", label) if w)
    return label == _CASE[expected](label)
