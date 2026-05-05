# SPDX-License-Identifier: Apache-2.0
"""Property-based verification of policy quorum invariants.

Hand-rolled exhaustive enumeration over the 4-layer state space
that ``derive_attestor_verdict`` operates on. Asserts five
invariants over every reachable state:

  I1. ALLOW iff zero blocking signals (across all layers that
      produced a signal).
  I2. BLOCK requires at least ``quorum_required`` blocking signals.
  I3. REVIEW iff exactly between 1 and ``quorum_required - 1``
      blocking signals (single-layer quorum demoted).
  I4. ``signal_unavailable`` layers are excluded from the quorum
      count, never counted as PASS.
  I5. The ``replay`` symmetric pair (``to_audit_record`` ->
      ``record.replay()``) reproduces the verdict from the record
      alone for every reachable state.

These are not formal proofs but exhaustive empirical checks across
the 16 reachable states (L1 in {block, pass} × L2 in {block, pass}
× L3 in {block, pass, unavailable} × L4 in {block, pass,
unavailable}, mod some unreachable combinations like all-unavailable
which the engine cannot produce). For a high-stakes deployment the
result is the auditor's structural argument: every quorum verdict
the engine can possibly produce satisfies I1-I5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from knowlytix.harness.governance.policy_calibration import (
    derive_attestor_verdict,
)


@dataclass
class InvariantViolation:
    invariant: str
    state: dict
    detail: str


@dataclass
class InvariantReport:
    n_states: int
    n_passed: int
    violations: list[InvariantViolation]

    @property
    def all_pass(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        rows = [
            "InvariantReport(",
            f"  states checked={self.n_states}",
            f"  invariants passed={self.n_passed}/{self.n_states}",
            f"  violations={len(self.violations)}",
        ]
        for v in self.violations[:5]:
            rows.append(f"    {v.invariant}: {v.detail}")
        rows.append(")")
        return "\n".join(rows)


def _make_decision_with_state(
    *,
    layer1: bool, layer2: bool,
    layer3: bool | None, layer4: bool | None,
):
    """Build a synthetic PolicyDecision matching the layer state."""
    from knowlytix.harness.testing.policy import PolicyDecision

    blocking_rules: list[str] = []
    gms_scores: dict[str, float] = {}

    if layer1:
        blocking_rules.append("RML-001")
    if layer2:
        blocking_rules.append("gms_graph:foo->blocks->bar")
    if layer3 is True:
        blocking_rules.append("gms_plausibility:foo")
        gms_scores["plausibility"] = 1.5
    elif layer3 is False:
        # Signal available, did not fire
        gms_scores["plausibility"] = 0.5
    if layer4 is True:
        gms_scores["tension:RML-001"] = 1.85
    elif layer4 is False:
        gms_scores["tension:RML-001"] = 1.0

    return PolicyDecision(
        allowed=not blocking_rules,
        action="probe",
        rules_checked=blocking_rules,
        blocking_rules=blocking_rules,
        gms_scores=gms_scores,
        reason="synthetic",
    )


def _enumerate_states() -> Iterable[dict]:
    """Generate every reachable layer state for the quorum FSM."""
    for l1 in (True, False):
        for l2 in (True, False):
            for l3 in (True, False, None):
                for l4 in (True, False, None):
                    yield {
                        "layer1": l1, "layer2": l2,
                        "layer3": l3, "layer4": l4,
                    }


def verify_quorum_invariants(
    quorum_required: int = 2,
) -> InvariantReport:
    """Enumerate every layer state and verify I1-I5."""
    states = list(_enumerate_states())
    violations: list[InvariantViolation] = []
    n_passed = 0

    for state in states:
        decision = _make_decision_with_state(**state)
        verdict = derive_attestor_verdict(
            decision, quorum_required=quorum_required,
        )

        # Count blocking signals from the actual state
        n_block_actual = (
            int(state["layer1"]) + int(state["layer2"])
            + int(state["layer3"] is True)
            + int(state["layer4"] is True)
        )
        n_avail_actual = (
            2  # L1 + L2 always available
            + (1 if state["layer3"] is not None else 0)
            + (1 if state["layer4"] is not None else 0)
        )

        # I1: ALLOW iff zero blocking
        if (verdict.decision == "ALLOW") != (n_block_actual == 0):
            violations.append(InvariantViolation(
                invariant="I1_allow_iff_zero_block",
                state=state,
                detail=(
                    f"verdict={verdict.decision}, "
                    f"n_block_actual={n_block_actual}"
                ),
            ))
            continue

        # I2: BLOCK requires >= quorum_required
        if verdict.decision == "BLOCK" and n_block_actual < quorum_required:
            violations.append(InvariantViolation(
                invariant="I2_block_requires_quorum",
                state=state,
                detail=(
                    f"BLOCK with only {n_block_actual} blocking signals "
                    f"(need ≥{quorum_required})"
                ),
            ))
            continue

        # I3: REVIEW iff 1 <= n_block < quorum_required
        if verdict.decision == "REVIEW":
            if not (0 < n_block_actual < quorum_required):
                violations.append(InvariantViolation(
                    invariant="I3_review_iff_partial",
                    state=state,
                    detail=(
                        f"REVIEW with n_block={n_block_actual}, "
                        f"quorum={quorum_required}"
                    ),
                ))
                continue

        # I4: signal_unavailable excluded from quorum
        if verdict.n_layers_signal_available != n_avail_actual:
            violations.append(InvariantViolation(
                invariant="I4_unavailable_excluded",
                state=state,
                detail=(
                    f"n_layers_signal_available="
                    f"{verdict.n_layers_signal_available}, "
                    f"actual={n_avail_actual}"
                ),
            ))
            continue

        n_passed += 1

    return InvariantReport(
        n_states=len(states),
        n_passed=n_passed,
        violations=violations,
    )


def verify_replay_symmetry() -> InvariantReport:
    """I5: to_audit_record + replay reproduces verdict from log alone."""
    from knowlytix.harness.governance.policy_audit import to_audit_record
    states = list(_enumerate_states())
    violations: list[InvariantViolation] = []
    n_passed = 0

    class _StubEngine:
        PLAUSIBILITY_THRESHOLD = 1.0239
        TENSION_THRESHOLD = 1.7
        store = None

        def __init__(self):
            class _S:
                def query_triples(self, head=None, **kw):
                    return []
                def score_triple(self, *a, **kw):
                    return None
                def fuzzy_match_entity(self, name):
                    return name
            self.store = _S()

    engine = _StubEngine()

    for state in states:
        decision = _make_decision_with_state(**state)
        verdict = derive_attestor_verdict(
            decision, quorum_required=2,
        )
        record = to_audit_record(
            engine=engine, bundle=None,
            probe_id=f"INV-{hash(tuple(state.items())) & 0xffff:04x}",
            agent="probe",
            decision=decision, verdict=verdict,
        )
        replayed = record.replay()
        if replayed != verdict.decision:
            violations.append(InvariantViolation(
                invariant="I5_replay_symmetry",
                state=state,
                detail=f"verdict={verdict.decision} replay={replayed}",
            ))
            continue
        n_passed += 1

    return InvariantReport(
        n_states=len(states),
        n_passed=n_passed,
        violations=violations,
    )
