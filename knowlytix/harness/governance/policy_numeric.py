# SPDX-License-Identifier: Apache-2.0
"""Numeric range index for state-tracking policy requirements.

Several responsible-ML rules reference numeric session state:

  * RML-007 ``session_budget_remaining`` -- must be > 0.
  * RML-004 ``min_two_models_trained``   -- must be ≥ 2.
  * RML-006 high-risk deployment may also reference threshold-style
    numeric requirements (e.g. AIR ≥ 0.80 in ADULT-003).

The plain Layer-1 check tells the engine *whether* the requirement
is satisfied (boolean). For a high-stakes deployment, that single
attestor is not enough: a value that just barely satisfies the
boolean check might still be far outside what the calibration
cohort observed, suggesting drift or a malformed runtime input.

This module produces a calibrated per-requirement
``(min, max, n_observations)`` range stamped into the policy bundle.
At decision time, the runtime computes a second attestor:

  * boolean PASS + value in calibrated range  ->  PASS
  * boolean PASS + value outside range        ->  REVIEW
  * boolean FAIL                              ->  BLOCK

Mirrors the pattern in
:mod:`governed_agent.governance_bundle::numeric_range_index` for
the orchestrator surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NumericObservation:
    """One labeled numeric observation from the calibration cohort."""

    requirement: str
    value: float
    accepted: bool   # whether this value satisfied the boolean check


@dataclass
class NumericRangeVerdict:
    """Outcome of the numeric two-attestor on one requirement."""

    requirement: str
    value: float
    boolean_pass: bool
    range_pass: bool
    range_min: float | None
    range_max: float | None
    n_observations: int
    decision: str       # PASS | REVIEW | BLOCK
    evidence: str


def build_numeric_range_index(
    observations: Iterable[NumericObservation],
) -> dict[str, tuple[float, float, int]]:
    """Aggregate observations into a per-requirement (min, max, n) index.

    Only ``accepted=True`` observations contribute to the calibrated
    range; rejected observations are counted but excluded from the
    range bounds (so a malformed value at calibration time does not
    poison the range).
    """
    by_req: dict[str, list[float]] = {}
    n_by_req: dict[str, int] = {}
    for obs in observations:
        n_by_req[obs.requirement] = n_by_req.get(obs.requirement, 0) + 1
        if obs.accepted:
            by_req.setdefault(obs.requirement, []).append(obs.value)

    index: dict[str, tuple[float, float, int]] = {}
    for req, values in by_req.items():
        if not values:
            continue
        index[req] = (
            float(min(values)),
            float(max(values)),
            int(n_by_req[req]),
        )
    return index


def check_numeric_range(
    requirement: str,
    value: float,
    boolean_pass: bool,
    index: dict[str, tuple[float, float, int]] | None,
    *,
    margin: float = 0.0,
) -> NumericRangeVerdict:
    """Two-attestor decision on a state-tracking requirement.

    ``margin`` widens the calibrated range by ``±margin`` to admit
    minor near-boundary drift without raising a REVIEW.
    """
    if not boolean_pass:
        return NumericRangeVerdict(
            requirement=requirement,
            value=value,
            boolean_pass=False,
            range_pass=False,
            range_min=None, range_max=None,
            n_observations=0,
            decision="BLOCK",
            evidence=(
                f"requirement '{requirement}' boolean check failed; "
                f"value={value} (Layer-1 BLOCK)"
            ),
        )

    if index is None or requirement not in index:
        return NumericRangeVerdict(
            requirement=requirement,
            value=value,
            boolean_pass=True,
            range_pass=True,
            range_min=None, range_max=None,
            n_observations=0,
            decision="PASS",
            evidence=(
                f"requirement '{requirement}' boolean check passed; "
                f"no calibrated range index available"
            ),
        )

    lo, hi, n = index[requirement]
    range_pass = (lo - margin) <= value <= (hi + margin)
    if range_pass:
        decision = "PASS"
        evidence = (
            f"value={value} in calibrated range "
            f"[{lo}-{margin}, {hi}+{margin}] (n={n})"
        )
    else:
        decision = "REVIEW"
        evidence = (
            f"value={value} OUTSIDE calibrated range [{lo}, {hi}] "
            f"(n={n}); demote to REVIEW"
        )

    return NumericRangeVerdict(
        requirement=requirement,
        value=value,
        boolean_pass=True,
        range_pass=range_pass,
        range_min=lo, range_max=hi,
        n_observations=n,
        decision=decision,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Default Adult Income observations
# ---------------------------------------------------------------------------


@dataclass
class NumericObservationLog:
    """Append-only telemetry log of numeric requirement observations.

    A deployment owns this artefact: every time the runtime
    encounters a state-tracking requirement, it appends a
    ``NumericObservation`` to the log. After ≥30 real observations
    accumulate per requirement, the deployment rebuilds the bundle's
    numeric_range_index from the log instead of the synthetic
    defaults shipped with the chapter.

    The log is JSON for diff-friendliness in version control;
    production deployments can swap the storage to Parquet or a
    time-series DB without changing the API.
    """

    path: str | None = None
    observations: list[NumericObservation] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.observations is None:
            self.observations = []

    def append(self, obs: NumericObservation) -> None:
        self.observations.append(obs)

    def append_many(self, obs: list[NumericObservation]) -> None:
        self.observations.extend(obs)

    def per_requirement_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.observations:
            counts[o.requirement] = counts.get(o.requirement, 0) + 1
        return counts

    def is_ready_for(self, requirement: str, *, min_n: int = 30) -> bool:
        """True when the log has enough observations to displace synthetic defaults."""
        return self.per_requirement_counts().get(requirement, 0) >= min_n

    def build_index(
        self, *, min_n: int = 30,
    ) -> dict[str, tuple[float, float, int]]:
        """Build a per-requirement (min, max, n) index from the log.

        Only requirements with ≥ ``min_n`` observations are
        included; those below the floor fall through to the
        synthetic defaults at runtime.
        """
        ready = [
            req for req, n in self.per_requirement_counts().items()
            if n >= min_n
        ]
        keep = [o for o in self.observations if o.requirement in ready]
        return build_numeric_range_index(keep)

    def merge_with_synthetic(
        self,
        synthetic: dict[str, tuple[float, float, int]],
        *,
        min_n: int = 30,
    ) -> dict[str, tuple[float, float, int]]:
        """Compose: real telemetry where ready, synthetic everywhere else.

        The deployment ships this merged index in the bundle. As
        more telemetry accumulates, fewer requirements fall back to
        synthetic, and the bundle becomes increasingly grounded in
        production data.
        """
        merged = dict(synthetic)
        merged.update(self.build_index(min_n=min_n))
        return merged

    def save(self, path: str | None = None) -> None:
        import json
        from pathlib import Path
        target = Path(path or self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            for o in self.observations:
                f.write(json.dumps({
                    "requirement": o.requirement,
                    "value": o.value,
                    "accepted": o.accepted,
                }) + "\n")

    @classmethod
    def load(cls, path: str) -> "NumericObservationLog":
        import json
        out: list[NumericObservation] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(NumericObservation(
                    requirement=d["requirement"],
                    value=float(d["value"]),
                    accepted=bool(d["accepted"]),
                ))
        return cls(path=path, observations=out)


def adult_income_numeric_observations() -> list[NumericObservation]:
    """Synthetic but plausible observations for the default cohort.

    These document what the bundle considers "normal range" for each
    state-tracking requirement. A deployment overrides this with its
    own production telemetry once it has accumulated 30+ observations.
    """
    obs: list[NumericObservation] = []
    # session_budget_remaining: typical session consumes 5-22 of 30
    # tool calls before requesting deploy
    for v in [3, 5, 8, 11, 14, 17, 20, 22, 25, 28]:
        obs.append(NumericObservation(
            requirement="session_budget_remaining",
            value=float(v), accepted=True,
        ))
    # rejected: zero or negative budget
    for v in [0, -1]:
        obs.append(NumericObservation(
            requirement="session_budget_remaining",
            value=float(v), accepted=False,
        ))

    # min_two_models_trained: leaderboard requires >=2 models
    for v in [2, 3, 4, 5, 6]:
        obs.append(NumericObservation(
            requirement="min_two_models_trained",
            value=float(v), accepted=True,
        ))
    for v in [0, 1]:
        obs.append(NumericObservation(
            requirement="min_two_models_trained",
            value=float(v), accepted=False,
        ))

    # adult_income AIR threshold (>=0.80)
    for v in [0.81, 0.83, 0.85, 0.87, 0.90, 0.93, 0.95]:
        obs.append(NumericObservation(
            requirement="air_above_0.80",
            value=float(v), accepted=True,
        ))
    for v in [0.65, 0.72, 0.78]:
        obs.append(NumericObservation(
            requirement="air_above_0.80",
            value=float(v), accepted=False,
        ))

    return obs
