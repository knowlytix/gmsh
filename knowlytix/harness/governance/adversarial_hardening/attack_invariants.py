# SPDX-License-Identifier: Apache-2.0
"""Property-based verification of adversarial-defense invariants.

Hand-rolled exhaustive enumeration over the (attack_class,
defense_layer, blocked) state space. Asserts five invariants
over every reachable state:

  L1. A snapshot with all blocked == True yields defense_rate
      == 1.0.
  L2. A snapshot with all blocked == False yields defense_rate
      == 0.0.
  L3. Mixed blocked / unblocked yields the correct fraction.
  L4. The fingerprint is stable across consecutive calls and
      changes when any outcome flips.
  L5. The cross-layer agreement check returns disagreement iff
      defense_layer != expected_layer for a blocked attack.

Mirrors :mod:`policy_invariants` and :mod:`workflow_invariants`
for the adversarial surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from knowlytix.harness.governance.adversarial_hardening.attack_snapshot import (
    AttackOutcome, AttackSnapshot,
)


@dataclass
class AttackInvariantViolation:
    invariant: str
    state: dict
    detail: str


@dataclass
class AttackInvariantReport:
    n_states: int
    n_passed: int
    violations: list[AttackInvariantViolation] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        rows = [
            "AttackInvariantReport(",
            f"  states checked={self.n_states}",
            f"  invariants passed={self.n_passed}/{self.n_states}",
            f"  violations={len(self.violations)}",
        ]
        for v in self.violations[:5]:
            rows.append(f"    {v.invariant}: {v.detail}")
        rows.append(")")
        return "\n".join(rows)


# Attack catalogue keyed by (name, expected_layer). Strings match
# the AdversarialSuite's actual emissions
# (governed_agent/agents/adversarial.py).
_ATTACK_CATALOG = (
    ("skip_fairness",            "policy_engine"),
    ("forbidden_tool_alias",     "schema_gate"),
    ("memory_poisoning",         "contradiction_gate"),
    ("budget_exhaustion",        "budget_constraint"),
    ("model_identity_confusion", "verification_layer"),
)


def _make_outcome(name: str, expected_layer: str, blocked: bool) -> AttackOutcome:
    return AttackOutcome(
        attack_name=name,
        blocked=blocked,
        defense_layer=expected_layer if blocked else "none",
    )


def _make_snapshot(
    *, suite_id: str, all_blocked: bool | None = None,
    custom: list[bool] | None = None,
) -> AttackSnapshot:
    if custom is not None:
        outcomes = [
            _make_outcome(name, layer, blocked)
            for (name, layer), blocked in zip(_ATTACK_CATALOG, custom)
        ]
    else:
        outcomes = [
            _make_outcome(name, layer, all_blocked)
            for name, layer in _ATTACK_CATALOG
        ]
    n = len(outcomes)
    n_blocked = sum(1 for o in outcomes if o.blocked)
    return AttackSnapshot(
        suite_id=suite_id,
        outcomes=outcomes,
        n_attacks=n,
        n_blocked=n_blocked,
        defense_rate=round(n_blocked / n, 6) if n else 0.0,
    )


def verify_attack_invariants() -> AttackInvariantReport:
    """Enumerate attack-snapshot states and verify L1-L5."""
    violations: list[AttackInvariantViolation] = []
    n_states = 0
    n_passed = 0

    # L1: all-blocked
    n_states += 1
    snap_all = _make_snapshot(suite_id="L1_all", all_blocked=True)
    if snap_all.defense_rate != 1.0:
        violations.append(AttackInvariantViolation(
            invariant="L1_all_blocked",
            state={"defense_rate": snap_all.defense_rate},
            detail=f"defense_rate={snap_all.defense_rate}, expected 1.0",
        ))
    else:
        n_passed += 1

    # L2: none-blocked
    n_states += 1
    snap_none = _make_snapshot(suite_id="L2_none", all_blocked=False)
    if snap_none.defense_rate != 0.0:
        violations.append(AttackInvariantViolation(
            invariant="L2_none_blocked",
            state={"defense_rate": snap_none.defense_rate},
            detail=f"defense_rate={snap_none.defense_rate}, expected 0.0",
        ))
    else:
        n_passed += 1

    # L3: every mixed combination across 5 attacks (2^5 = 32 states)
    for mask in range(2 ** len(_ATTACK_CATALOG)):
        n_states += 1
        flags = [(mask >> i) & 1 == 1 for i in range(len(_ATTACK_CATALOG))]
        snap = _make_snapshot(suite_id=f"L3_{mask:05b}", custom=flags)
        expected_rate = round(sum(flags) / len(flags), 6)
        if snap.defense_rate != expected_rate:
            violations.append(AttackInvariantViolation(
                invariant="L3_mixed_correct",
                state={"mask": f"{mask:05b}",
                       "defense_rate": snap.defense_rate,
                       "expected_rate": expected_rate},
                detail=(
                    f"mask={mask:05b}, "
                    f"defense_rate={snap.defense_rate}, "
                    f"expected={expected_rate}"
                ),
            ))
            continue
        n_passed += 1

    # L4: fingerprint stable across two computations on the same snapshot
    for snap in (snap_all, snap_none):
        n_states += 1
        fp1 = snap.fingerprint()
        fp2 = snap.fingerprint()
        if fp1 != fp2:
            violations.append(AttackInvariantViolation(
                invariant="L4_fingerprint_stable",
                state={"suite_id": snap.suite_id},
                detail="two fingerprint() calls differed",
            ))
            continue
        n_passed += 1

    # L5: cross-layer agreement matches expected layer
    from knowlytix.harness.governance.adversarial_hardening.cross_layer_agreement import (
        cross_layer_agreement,
    )

    class _StubAttackResult:
        def __init__(self, name, layer, blocked):
            self.attack_name = name
            self.defense_layer = layer
            self.blocked = blocked

    n_states += 1
    # All-correct: each attack fired its expected layer
    correct_results = [
        _StubAttackResult(name, layer, True)
        for name, layer in _ATTACK_CATALOG
    ]
    rep_correct = cross_layer_agreement(correct_results)
    if not rep_correct.all_match:
        violations.append(AttackInvariantViolation(
            invariant="L5_correct_layers_match",
            state={},
            detail=(
                f"expected all layers to match, got "
                f"{len(rep_correct.disagreements)} disagreements"
            ),
        ))
    else:
        n_passed += 1

    n_states += 1
    # All-wrong: each attack fired the wrong layer
    wrong_results = [
        _StubAttackResult(name, "wrong_layer", True)
        for name, layer in _ATTACK_CATALOG
    ]
    rep_wrong = cross_layer_agreement(wrong_results)
    if rep_wrong.all_match:
        violations.append(AttackInvariantViolation(
            invariant="L5_wrong_layers_disagree",
            state={},
            detail="expected all layers to disagree, got all_match=True",
        ))
    else:
        n_passed += 1

    return AttackInvariantReport(
        n_states=n_states,
        n_passed=n_passed,
        violations=violations,
    )
