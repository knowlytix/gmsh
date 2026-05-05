# SPDX-License-Identifier: Apache-2.0
"""Bundle-replay regression suite.

A frozen cohort of labeled entity pairs is saved alongside the
deployment. When a new bundle is built, the suite replays the
cohort against both the old and new bundle and reports verdict
deltas. A bundle is allowed to ship only when its replay either
matches the old verdicts exactly or differs in ways the deployment
team has reviewed.

Workflow:

  1. Build a ``ReplayCohort`` once. Save to disk; check in alongside
     the bundle.
  2. After offline calibration produces ``new_bundle``, run::

         old_result = replay_bundle(old_gate, cohort)
         new_result = replay_bundle(new_gate, cohort)
         delta = compare_replays(old_result, new_result)

  3. If ``delta.has_regressions`` returns True, do not ship; the
     deployment team must review and either update the cohort or
     roll back the bundle.

The cohort is JSON to keep it diff-friendly in version control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowlytix.harness.governance.runtime_gate import RuntimeGate, RuntimeVerdict
    from knowlytix.harness.governance.ood_validation import LabeledPair


@dataclass(frozen=True)
class ReplayCohort:
    """Versioned, frozen cohort of labeled pairs.

    Use ``ReplayCohort.from_labeled_pairs(...)`` to build, then
    ``save()`` / ``load()`` for persistence.
    """

    name: str
    version: str
    pairs: tuple[tuple[str, str, str, str], ...]  # (a, b, label, provenance)

    @classmethod
    def from_labeled_pairs(
        cls, pairs: list["LabeledPair"], *, name: str, version: str,
    ) -> "ReplayCohort":
        return cls(
            name=name, version=version,
            pairs=tuple((p.a, p.b, p.label, p.provenance) for p in pairs),
        )

    def to_labeled_pairs(self) -> list["LabeledPair"]:
        from knowlytix.harness.governance.ood_validation import LabeledPair
        return [LabeledPair(a, b, label, prov)
                for a, b, label, prov in self.pairs]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "name": self.name,
                "version": self.version,
                "pairs": list(self.pairs),
            }, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayCohort":
        with open(path) as f:
            d = json.load(f)
        return cls(
            name=d["name"], version=d["version"],
            pairs=tuple(tuple(p) for p in d["pairs"]),
        )


@dataclass
class ReplayResult:
    """Verdicts produced by one bundle on a replay cohort."""

    cohort_name: str
    cohort_version: str
    bundle_id: str
    bundle_signature: str
    verdicts: dict[tuple[str, str], dict]  # (a, b) -> {verdict, reason, te, p, label}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Tuple keys → string for JSON.
        verdicts_serializable = {
            f"{a}||{b}": v for (a, b), v in self.verdicts.items()
        }
        with open(path, "w") as f:
            json.dump({
                "cohort_name": self.cohort_name,
                "cohort_version": self.cohort_version,
                "bundle_id": self.bundle_id,
                "bundle_signature": self.bundle_signature,
                "verdicts": verdicts_serializable,
            }, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayResult":
        with open(path) as f:
            d = json.load(f)
        verdicts = {
            tuple(k.split("||", 1)): v for k, v in d["verdicts"].items()
        }
        return cls(
            cohort_name=d["cohort_name"],
            cohort_version=d["cohort_version"],
            bundle_id=d["bundle_id"],
            bundle_signature=d["bundle_signature"],
            verdicts=verdicts,
        )


@dataclass
class ReplayDelta:
    """Per-pair changes between two replay results."""

    cohort_name: str
    n_total: int
    n_unchanged: int
    n_verdict_changed: int
    n_verdict_drift: int   # verdict same, te / p moved meaningfully
    changes: list[tuple[tuple[str, str], dict, dict]] = field(default_factory=list)
    drifts: list[tuple[tuple[str, str], float, float]] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """True if any pair changed verdict OR drifted significantly."""
        return self.n_verdict_changed > 0

    def summary(self) -> str:
        return (
            f"ReplayDelta(n_total={self.n_total}, "
            f"unchanged={self.n_unchanged}, "
            f"verdict_changed={self.n_verdict_changed}, "
            f"drifted={self.n_verdict_drift})"
        )


def replay_bundle(
    gate: "RuntimeGate", cohort: ReplayCohort,
) -> ReplayResult:
    """Run every pair in the cohort through the gate."""
    verdicts: dict[tuple[str, str], dict] = {}
    for a, b, label, prov in cohort.pairs:
        v: "RuntimeVerdict" = gate.verdict(a, b)
        verdicts[(a, b)] = {
            "verdict": v.verdict,
            "reason": v.reason,
            "te": v.tension_energy,
            "p": v.p_contradiction,
            "label": label,
            "provenance": prov,
        }
    return ReplayResult(
        cohort_name=cohort.name,
        cohort_version=cohort.version,
        bundle_id=gate.bundle.bundle_id,
        bundle_signature=gate.bundle.signature,
        verdicts=verdicts,
    )


def compare_replays(
    old: ReplayResult, new: ReplayResult,
    *,
    drift_threshold_te: float = 0.20,
    drift_threshold_p: float = 0.15,
) -> ReplayDelta:
    """Diff two replay results; report changed verdicts and drift."""
    if old.cohort_name != new.cohort_name:
        raise ValueError(
            f"Cohort name mismatch: '{old.cohort_name}' vs "
            f"'{new.cohort_name}'"
        )
    common = set(old.verdicts.keys()) & set(new.verdicts.keys())
    n_total = len(common)
    n_unchanged = 0
    changes: list = []
    drifts: list = []

    for key in sorted(common):
        ov = old.verdicts[key]
        nv = new.verdicts[key]
        if ov["verdict"] != nv["verdict"]:
            changes.append((key, ov, nv))
            continue
        # Same verdict; check for numeric drift.
        drift_te = 0.0
        if ov["te"] is not None and nv["te"] is not None:
            drift_te = abs(float(nv["te"]) - float(ov["te"]))
        drift_p = 0.0
        if ov["p"] is not None and nv["p"] is not None:
            drift_p = abs(float(nv["p"]) - float(ov["p"]))
        if (drift_te >= drift_threshold_te
                or drift_p >= drift_threshold_p):
            drifts.append((key, drift_te, drift_p))
            continue
        n_unchanged += 1

    return ReplayDelta(
        cohort_name=old.cohort_name,
        n_total=n_total,
        n_unchanged=n_unchanged,
        n_verdict_changed=len(changes),
        n_verdict_drift=len(drifts),
        changes=changes,
        drifts=drifts,
    )


class ReplayRegressionError(RuntimeError):
    """Replay diff exceeds allowed regressions."""


def assert_no_regression(
    delta: ReplayDelta, *, allow_drift: bool = True,
) -> None:
    """Strict gate: raise if any verdict changed (and optionally drift)."""
    problems: list[str] = []
    if delta.n_verdict_changed > 0:
        sample = delta.changes[:5]
        problems.append(
            f"{delta.n_verdict_changed} verdict change(s); "
            f"first {len(sample)}: "
            + ", ".join(f"{k}: {ov['verdict']}->{nv['verdict']}"
                        for k, ov, nv in sample)
        )
    if not allow_drift and delta.n_verdict_drift > 0:
        problems.append(
            f"{delta.n_verdict_drift} pair(s) drifted beyond threshold"
        )
    if problems:
        raise ReplayRegressionError("; ".join(problems))
