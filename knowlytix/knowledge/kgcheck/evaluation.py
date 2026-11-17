# SPDX-License-Identifier: Apache-2.0
"""Measure a semantic reviewer (EVD005) and the heuristic rules on labeled data.

A reviewer's verdicts are advisory until measured. ``evaluate_reviewer`` runs a
reviewer on labeled claims and reports, per gold label, precision and recall
of its committed verdicts, its abstention rate and the confusion matrix, on a
held-out split of documents when one is given. Claims are split by document,
never by claim, so passages of one document cannot appear on both sides.

``calibrate_rule`` does the same for a heuristic rule parameter: on training
documents it picks the value with the highest recall whose precision reaches
the target, then reports precision and recall of that value on the held-out
documents. The value is promoted into a profile only when the lower 95%
Wilson bound of the held-out precision reaches the target (plan section 10.3:
0.95), so a handful of correct held-out findings cannot promote it; otherwise
the rule keeps its shipped parameters. Labels come from people; this module never
produces them. ``labeling_sheet`` exports a report's findings for adjudication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from knowlytix.knowledge.kgcheck.config import AuditConfig
from knowlytix.knowledge.kgcheck.engine import audit
from knowlytix.knowledge.kgcheck.frozen import thaw
from knowlytix.knowledge.kgcheck.models import AuditReport, AuditSnapshot, JsonObject
from knowlytix.knowledge.kgcheck.review import ABSTAIN, LABELS, SemanticReviewer

TARGET_PRECISION = 0.95


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


# --- semantic review ----------------------------------------------------------------
@dataclass(frozen=True)
class LabeledClaim:
    """A fact, the passages cited for it, and its gold label (one of ``review.LABELS``)."""

    claim_id: str
    document: str
    fact: tuple[str, str, str]
    passages: tuple[str, ...]
    label: str
    note: str = ""  # how the label was obtained, e.g. "expert" or "constructed: value changed"

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")


@dataclass(frozen=True)
class ReviewEvaluation:
    """Verdicts against gold labels on one split."""

    split: str
    claims: int
    confusion: Mapping[str, Mapping[str, int]]  # gold -> verdict -> count
    per_label: Mapping[str, Mapping[str, float | int | None]]
    abstention_rate: float | None
    accuracy_committed: float | None
    model: str
    prompt_version: str
    verdicts: tuple[tuple[str, str], ...] = field(default=())  # (claim id, verdict)

    def to_dict(self) -> JsonObject:
        return {"split": self.split, "claims": self.claims, "model": self.model,
                "prompt_version": self.prompt_version,
                "confusion": {g: dict(v) for g, v in self.confusion.items()},
                "per_label": {g: dict(v) for g, v in self.per_label.items()},
                "abstention_rate": self.abstention_rate,
                "accuracy_committed": self.accuracy_committed,
                "verdicts": [list(v) for v in self.verdicts]}


def evaluate_reviewer(reviewer: SemanticReviewer, claims: Iterable[LabeledClaim], *,
                      held_out: Iterable[str] | None = None) -> dict[str, ReviewEvaluation]:
    """Evaluations keyed ``all``, and ``train`` and ``held_out`` when documents are held out."""
    from knowlytix.knowledge.kgcheck.review import PROMPT_VERSION
    claims = list(claims)
    ids = [c.claim_id for c in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("claim ids must be unique")
    verdicts = {c.claim_id: reviewer.review(c.fact, c.passages).label for c in claims}
    splits = {"all": claims}
    if held_out is not None:
        documents = set(held_out)
        splits["held_out"] = [c for c in claims if c.document in documents]
        splits["train"] = [c for c in claims if c.document not in documents]
    return {name: _evaluate(name, subset, verdicts, reviewer.model, PROMPT_VERSION)
            for name, subset in splits.items()}


def _evaluate(split: str, claims: Sequence[LabeledClaim], verdicts: Mapping[str, str],
              model: str, prompt: str) -> ReviewEvaluation:
    confusion = {g: {v: 0 for v in (*LABELS, ABSTAIN)} for g in LABELS}
    for c in claims:
        confusion[c.label][verdicts[c.claim_id]] += 1
    per_label = {}
    for label in LABELS:
        predicted = sum(confusion[g][label] for g in LABELS)
        correct = confusion[label][label]
        gold = sum(confusion[label].values())
        per_label[label] = {"gold": gold, "predicted": predicted,
                            "precision": _ratio(correct, predicted), "recall": _ratio(correct, gold)}
    abstained = sum(confusion[g][ABSTAIN] for g in LABELS)
    committed = len(claims) - abstained
    return ReviewEvaluation(split, len(claims), confusion, per_label, _ratio(abstained, len(claims)),
                            _ratio(sum(confusion[g][g] for g in LABELS), committed), model, prompt,
                            tuple(sorted((c.claim_id, verdicts[c.claim_id]) for c in claims)))


# --- heuristic rule calibration ---------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    """The chosen parameter value of one rule and its precision and recall on each split."""

    rule_id: str
    parameter: str
    value: object | None
    train: Mapping[str, float | int | None]
    held_out: Mapping[str, float | int | None]
    target_precision: float
    promote: bool
    by_value: tuple[tuple[object, Mapping[str, float | int | None]], ...]

    def to_dict(self) -> JsonObject:
        return {"rule_id": self.rule_id, "parameter": self.parameter, "value": self.value,
                "train": dict(self.train), "held_out": dict(self.held_out),
                "target_precision": self.target_precision, "promote": self.promote,
                "by_value": [[v, dict(m)] for v, m in self.by_value]}


Labels = Mapping[tuple[str, str], bool]  # (document, finding key) -> is a true issue


def finding_key(finding: object) -> str:
    """What a label is attached to: the rule and the affected ids, independent of parameters."""
    return f"{finding.rule_id}|{'|'.join(sorted(finding.affected))}|{finding.discriminator}"  # type: ignore[attr-defined]


def calibrate_rule(rule_id: str, parameter: str, values: Sequence[object],
                   snapshots: Mapping[str, AuditSnapshot], labels: Labels, *,
                   held_out: Iterable[str], target_precision: float = TARGET_PRECISION,
                   run: Callable[[AuditSnapshot, AuditConfig], AuditReport] = audit) -> Calibration:
    """Choose ``parameter`` of ``rule_id`` on the training documents; measure it held out.

    Positives are the labeled true issues; a finding without a label counts as a
    false positive, so unlabeled output never inflates precision.
    """
    documents = set(held_out)
    by_value = []
    for value in values:
        # every finding must be listed: a truncated list would undercount recall
        config = AuditConfig(rules=(rule_id,), params={rule_id: {parameter: value}},
                             max_findings=10**9)
        found = {doc: _keys(run(snap, config)) for doc, snap in snapshots.items()}
        by_value.append((value, _scores(found, labels, lambda d: d not in documents),
                         _scores(found, labels, lambda d: d in documents)))
    eligible = [(v, tr, ho) for v, tr, ho in by_value
                if tr["precision"] is not None and tr["precision"] >= target_precision]
    best = max(eligible, key=lambda x: (x[1]["recall"] or 0, x[1]["precision"]), default=None)
    value, train, test = best if best else (None, {}, {})
    lower = _wilson_lower(test.get("true_positives", 0),
                          test.get("true_positives", 0) + test.get("false_positives", 0))
    test = {**test, "precision_lower_bound": lower}
    promote = bool(best) and lower is not None and lower >= target_precision
    return Calibration(rule_id, parameter, value, train, test, target_precision, promote,
                       tuple((v, {"train_" + k: x for k, x in tr.items()} |
                              {"held_out_" + k: x for k, x in ho.items()}) for v, tr, ho in by_value))


def _keys(report: AuditReport) -> set[str]:
    if any(r.details_truncated for r in report.results):
        raise ValueError("calibration needs every finding listed; a result was truncated")
    return {finding_key(f) for f in report.findings()}


def _wilson_lower(successes: int, n: int, z: float = 1.959964) -> float | None:
    """Lower bound of the Wilson score interval for a proportion (95% by default)."""
    if n == 0:
        return None
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return round((centre - margin) / (1 + z * z / n), 4)


def _scores(found: Mapping[str, set[str]], labels: Labels,
            keep: Callable[[str], bool]) -> dict[str, float | int | None]:
    tp = fp = 0
    for doc, keys in found.items():
        if not keep(doc):
            continue
        for key in keys:
            if labels.get((doc, key)) is True:
                tp += 1
            else:
                fp += 1
    positives = sum(1 for (doc, _), ok in labels.items() if ok and keep(doc))
    return {"true_positives": tp, "false_positives": fp, "positives": positives,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, positives)}


def labeling_sheet(report: AuditReport, document: str, rules: Iterable[str] | None = None
                   ) -> list[JsonObject]:
    """One row per finding to adjudicate: its key, message and witness, and an empty label."""
    wanted = set(rules) if rules is not None else None
    return [{"document": document, "key": finding_key(f), "rule_id": f.rule_id,
             "message": f.message, "witness": thaw(f.witness), "label": None}
            for f in report.findings() if wanted is None or f.rule_id in wanted]


def calibrated_profile(base: Mapping[str, object], calibrations: Iterable[Calibration],
                       name: str) -> JsonObject:
    """A profile like ``base`` with the parameter of every promoted calibration set and
    the rule enabled; its version records the calibration.

    Rules whose calibration is not promoted are left as they are in ``base``.
    """
    params = {k: dict(v) for k, v in dict(base.get("params", {})).items()}  # type: ignore[arg-type]
    rules = list(base.get("rules", ()))  # type: ignore[call-overload]
    promoted = []
    for c in calibrations:
        if c.promote:
            params.setdefault(c.rule_id, {})[c.parameter] = c.value
            promoted.append(c.rule_id)
            if c.rule_id not in rules:
                rules.append(c.rule_id)
    return {**dict(base), "profile": name, "params": params, "rules": rules,
            "version": f"{base.get('version', '0')}+calibrated",
            "description": f"{base.get('description', '')} Calibrated parameters for "
                           f"{', '.join(sorted(promoted)) or 'no rule'}."}
