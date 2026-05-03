# SPDX-License-Identifier: Apache-2.0
"""Systematic adversarial cohort for audit-trail hardening.

The hash chain (\\Cref{hash_chain}) detects tampering and the
cross-reference attestor (\\Cref{cross_attestor}) detects gaps,
but neither has been exercised against a written-down set of
attacks.  A claim that "the chain catches everything" without a
regression battery is the same kind of unverified assertion the
chapter is supposed to prevent.

This module generates a cohort of audit-trail tampers, each with a
documented expected-detection contract: the chain or the
cross-reference attestor (or both) must report the break, and the
report must mention the kind of attack.  A probe that escapes
detection or that fires the wrong attestor is a regression.

Probe types:

* ``content_edit``     --- mutate a field in one packet (block/allow
  flip, rule deletion, timestamp rewrite).  Hash chain must report
  ``content``.
* ``insert_forged``    --- insert a synthetic packet between two
  legitimate ones.  Chain must report ``missing_hash`` (the
  forgery has no hash) or ``prev`` (downstream chain breaks).
* ``truncate_middle``  --- remove a packet from the interior.
  Chain must report ``prev`` on the next packet.
* ``reorder``          --- swap two packets in the trail.  Chain
  must report ``prev`` (the broken link).
* ``unsealed``         --- forget to seal.  Chain must report
  ``missing_hash`` on every packet.
* ``runtime_drop``     --- runtime recorded a blocked decision but
  the audit trail does not include the matching packet.
  Cross-attestor must report ``missing_packet``.
* ``runtime_extra``    --- audit trail has a packet the runtime
  never recorded.  Cross-attestor must report ``extra_packet``.
* ``decision_flip``    --- audit packet's ``policy_decision`` is
  flipped from the runtime's.  Cross-attestor must report
  ``decision_mismatch``.
* ``rules_swap``       --- audit packet has different blocking
  rules than the runtime.  Cross-attestor must report
  ``rules_mismatch``.

The probe set scales by attack family, not by hand-curated
imagination.  Adding a future attestor is a matter of registering
new probes here and asserting the report names them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

from knowlytix.harness.governance.tracing import AuditPacket
from knowlytix.harness.governance.tracing_hardening.hash_chain import (
    seal_audit_trail, verify_audit_chain,
)
from knowlytix.harness.governance.tracing_hardening.cross_attestor import (
    RuntimePolicyEvent,
    cross_attest_policy,
)


@dataclass(frozen=True)
class AuditAdversarialProbe:
    """One systematic probe over the hardened audit trail.

    ``mutator`` produces the (tampered_packets, runtime_events)
    pair from a fresh sealed baseline.  ``expected_attestor`` is
    one of ``"chain"`` / ``"cross"`` / ``"both"``.
    ``expected_kinds`` lists the kinds the named attestor must
    report (case-sensitive, exact match).
    """

    name: str
    probe_type: str
    expected_attestor: str
    expected_kinds: tuple[str, ...]
    mutator: Callable[
        [list[AuditPacket], list[RuntimePolicyEvent]],
        tuple[list[AuditPacket], list[RuntimePolicyEvent]],
    ]


@dataclass
class AuditAdversarialReport:
    n_total: int
    per_kind_n: dict[str, int]
    per_kind_safe: dict[str, int]
    per_kind_unsafe: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict
    )

    @property
    def safety_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return sum(self.per_kind_safe.values()) / self.n_total

    def summary(self) -> str:
        rows = [
            f"n_total={self.n_total}",
            f"safety_rate={self.safety_rate:.3f}",
        ]
        for kind, n in sorted(self.per_kind_n.items()):
            safe = self.per_kind_safe.get(kind, 0)
            rate = safe / n if n else 0.0
            rows.append(f"  {kind}: {safe}/{n} safe ({rate:.2%})")
        return "AuditAdversarialReport(\n  " + "\n  ".join(rows) + "\n)"


# -------------------------------------------------------------------- #
# Baseline trail factory                                               #
# -------------------------------------------------------------------- #


def make_baseline_trail() -> tuple[list[AuditPacket], list[RuntimePolicyEvent]]:
    """Build a small but representative sealed trail and runtime log.

    Three allowed packets, one blocked.  Suffices to exercise every
    probe type without inflating the cohort.
    """
    packets = [
        AuditPacket(
            step_num=0, tool_name="load_dataset",
            policy_decision="allowed", blocking_rules=[],
            gms_scores={"plausibility": 0.8},
            timestamp=1000.0, state_before="idle", state_after="data_loaded",
        ),
        AuditPacket(
            step_num=1, tool_name="preprocess_data",
            policy_decision="allowed", blocking_rules=[],
            gms_scores={"plausibility": 0.7},
            timestamp=1001.0,
            state_before="data_loaded", state_after="preprocessed",
        ),
        AuditPacket(
            step_num=2, tool_name="delete_database",
            policy_decision="blocked", blocking_rules=["forbidden_tool"],
            gms_scores={"plausibility": 0.0},
            timestamp=1002.0,
            state_before="preprocessed", state_after="preprocessed",
        ),
        AuditPacket(
            step_num=3, tool_name="train_model",
            policy_decision="allowed", blocking_rules=[],
            gms_scores={"plausibility": 0.6},
            timestamp=1003.0,
            state_before="preprocessed", state_after="model_trained",
        ),
    ]
    seal_audit_trail(packets)

    rt_events = [
        RuntimePolicyEvent(0, "load_dataset", "allowed"),
        RuntimePolicyEvent(1, "preprocess_data", "allowed"),
        RuntimePolicyEvent(2, "delete_database", "blocked",
                           ("forbidden_tool",)),
        RuntimePolicyEvent(3, "train_model", "allowed"),
    ]
    return packets, rt_events


# -------------------------------------------------------------------- #
# Mutators                                                             #
# -------------------------------------------------------------------- #


def _content_edit(
    packets: list[AuditPacket],
    rt: list[RuntimePolicyEvent],
) -> tuple[list[AuditPacket], list[RuntimePolicyEvent]]:
    pkts = copy.deepcopy(packets)
    pkts[2].policy_decision = "allowed"
    pkts[2].blocking_rules = []
    return pkts, rt


def _insert_forged(packets, rt):
    pkts = copy.deepcopy(packets)
    fake = AuditPacket(
        step_num=99, tool_name="forged_after_chain",
        policy_decision="allowed",
        timestamp=999.0,
    )
    return pkts[:2] + [fake] + pkts[2:], rt


def _truncate_middle(packets, rt):
    pkts = copy.deepcopy(packets)
    return pkts[:2] + pkts[3:], rt


def _reorder(packets, rt):
    pkts = copy.deepcopy(packets)
    pkts[1], pkts[2] = pkts[2], pkts[1]
    return pkts, rt


def _unsealed(packets, rt):
    pkts = copy.deepcopy(packets)
    for p in pkts:
        p.content_hash = ""
        p.prev_hash = ""
    return pkts, rt


def _runtime_drop(packets, rt):
    pkts = copy.deepcopy(packets)
    return pkts[:2] + pkts[3:], rt  # drop the blocked packet


def _runtime_extra(packets, rt):
    pkts = copy.deepcopy(packets)
    fake = AuditPacket(
        step_num=99, tool_name="phantom_tool",
        policy_decision="allowed", timestamp=9999.0,
    )
    pkts.append(fake)
    seal_audit_trail(pkts)  # keep chain valid; cross-attestor catches it
    return pkts, rt


def _decision_flip(packets, rt):
    pkts = copy.deepcopy(packets)
    pkts[2].policy_decision = "allowed"
    pkts[2].blocking_rules = []
    seal_audit_trail(pkts)  # re-seal so chain is consistent; cross catches
    return pkts, rt


def _rules_swap(packets, rt):
    pkts = copy.deepcopy(packets)
    pkts[2].blocking_rules = ["wrong_rule"]
    seal_audit_trail(pkts)
    return pkts, rt


# -------------------------------------------------------------------- #
# Cohort builder                                                       #
# -------------------------------------------------------------------- #


def build_audit_adversarial_cohort() -> list[AuditAdversarialProbe]:
    """Assemble the v1 audit-trail adversarial cohort (one probe per attack class).

    Use :func:`build_audit_adversarial_cohort_v2` for the
    n_per_type-parameterised production cohort.
    """
    return [
        AuditAdversarialProbe(
            name="content_edit_block_to_allow",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("content",),
            mutator=_content_edit,
        ),
        AuditAdversarialProbe(
            name="insert_forged_packet",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("missing_hash", "prev"),
            mutator=_insert_forged,
        ),
        AuditAdversarialProbe(
            name="truncate_middle",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("prev",),
            mutator=_truncate_middle,
        ),
        AuditAdversarialProbe(
            name="reorder_two_packets",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("prev",),
            mutator=_reorder,
        ),
        AuditAdversarialProbe(
            name="unsealed_trail",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("missing_hash",),
            mutator=_unsealed,
        ),
        AuditAdversarialProbe(
            name="runtime_drop_blocked",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("missing_packet",),
            mutator=_runtime_drop,
        ),
        AuditAdversarialProbe(
            name="runtime_extra_phantom",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("extra_packet",),
            mutator=_runtime_extra,
        ),
        AuditAdversarialProbe(
            name="decision_flip_after_seal",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("decision_mismatch", "rules_mismatch"),
            mutator=_decision_flip,
        ),
        AuditAdversarialProbe(
            name="rules_swap_after_seal",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("rules_mismatch",),
            mutator=_rules_swap,
        ),
    ]


# -------------------------------------------------------------------- #
# Parameterised v2 cohort: position / index / payload variants         #
# -------------------------------------------------------------------- #


def _make_content_edit_at(idx: int):
    """Mutator factory: content edit at a specific packet index."""
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx >= len(pkts):
            return pkts, rt
        # Flip whatever decision is recorded so the test does not
        # implicitly assume the baseline trail's exact contents.
        if pkts[idx].policy_decision == "blocked":
            pkts[idx].policy_decision = "allowed"
            pkts[idx].blocking_rules = []
        else:
            pkts[idx].policy_decision = "blocked"
            pkts[idx].blocking_rules = ["forged_block"]
        return pkts, rt
    return _mut


def _make_timestamp_rewrite_at(idx: int):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx >= len(pkts):
            return pkts, rt
        pkts[idx].timestamp = pkts[idx].timestamp + 86400.0  # +1 day
        return pkts, rt
    return _mut


def _make_insert_forged_at(idx: int):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx > len(pkts):
            return pkts, rt
        fake = AuditPacket(
            step_num=900 + idx, tool_name=f"forged_at_{idx}",
            policy_decision="allowed", timestamp=900.0 + idx,
        )
        return pkts[:idx] + [fake] + pkts[idx:], rt
    return _mut


def _make_truncate_at(idx: int):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx >= len(pkts):
            return pkts, rt
        return pkts[:idx] + pkts[idx + 1:], rt
    return _mut


def _make_reorder(i: int, j: int):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if i >= len(pkts) or j >= len(pkts):
            return pkts, rt
        pkts[i], pkts[j] = pkts[j], pkts[i]
        return pkts, rt
    return _mut


def _make_runtime_drop_at(idx: int):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx >= len(pkts):
            return pkts, rt
        return pkts[:idx] + pkts[idx + 1:], rt
    return _mut


def _make_runtime_extra(name_suffix: str):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        fake = AuditPacket(
            step_num=99, tool_name=f"phantom_{name_suffix}",
            policy_decision="allowed", timestamp=9999.0,
        )
        pkts.append(fake)
        seal_audit_trail(pkts)
        return pkts, rt
    return _mut


def _make_decision_flip_at(idx: int):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx >= len(pkts):
            return pkts, rt
        if pkts[idx].policy_decision == "blocked":
            pkts[idx].policy_decision = "allowed"
            pkts[idx].blocking_rules = []
        else:
            pkts[idx].policy_decision = "blocked"
            pkts[idx].blocking_rules = ["forged"]
        seal_audit_trail(pkts)
        return pkts, rt
    return _mut


def _make_rules_swap_at(idx: int, replacement: tuple[str, ...]):
    def _mut(packets, rt):
        pkts = copy.deepcopy(packets)
        if idx >= len(pkts):
            return pkts, rt
        pkts[idx].blocking_rules = list(replacement)
        seal_audit_trail(pkts)
        return pkts, rt
    return _mut


def build_audit_adversarial_cohort_v2(
    *, n_per_type: int = 6,
) -> list[AuditAdversarialProbe]:
    """Production cohort: parameterised attacks at multiple positions.

    Each attack class produces ``n_per_type`` variants by sweeping
    the position / index / payload axis. Default ``n_per_type=6``
    yields ~54 probes.
    """
    cohort: list[AuditAdversarialProbe] = []

    # Indices to target across the 4-packet baseline trail
    base_indices = [0, 1, 2, 3]

    # 1. content_edit at every index
    for i in range(n_per_type):
        idx = base_indices[i % len(base_indices)]
        cohort.append(AuditAdversarialProbe(
            name=f"content_edit_at_{idx}_v{i}",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("content",),
            mutator=_make_content_edit_at(idx),
        ))

    # 2. timestamp_rewrite at every index — chain catches via content
    for i in range(n_per_type):
        idx = base_indices[i % len(base_indices)]
        cohort.append(AuditAdversarialProbe(
            name=f"timestamp_rewrite_at_{idx}_v{i}",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("content",),
            mutator=_make_timestamp_rewrite_at(idx),
        ))

    # 3. insert_forged at every position
    for i in range(n_per_type):
        idx = base_indices[i % len(base_indices)]
        cohort.append(AuditAdversarialProbe(
            name=f"insert_forged_at_{idx}_v{i}",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("missing_hash", "prev"),
            mutator=_make_insert_forged_at(idx),
        ))

    # 4. truncate at every position. Mid-trail truncation breaks the
    #    chain (next packet's prev_hash no longer matches); tail
    #    truncation is invisible to the chain alone (no successor to
    #    break) -- only the cross-attestor catches it via
    #    missing_packet. This is the structural reason both
    #    attestors are needed (defence in depth).
    for i in range(n_per_type):
        idx = base_indices[i % len(base_indices)]
        is_tail = (idx == len(base_indices) - 1)
        if is_tail:
            cohort.append(AuditAdversarialProbe(
                name=f"truncate_tail_{idx}_v{i}",
                probe_type="cross",
                expected_attestor="cross",
                expected_kinds=("missing_packet",),
                mutator=_make_truncate_at(idx),
            ))
        else:
            cohort.append(AuditAdversarialProbe(
                name=f"truncate_mid_{idx}_v{i}",
                probe_type="chain",
                expected_attestor="chain",
                expected_kinds=("prev",),
                mutator=_make_truncate_at(idx),
            ))

    # 5. reorder permutations
    swap_pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    for i in range(n_per_type):
        a, b = swap_pairs[i % len(swap_pairs)]
        cohort.append(AuditAdversarialProbe(
            name=f"reorder_{a}_{b}_v{i}",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("prev",),
            mutator=_make_reorder(a, b),
        ))

    # 6. unsealed_trail (parameter-free; one probe is sufficient
    #    but we keep n_per_type by adding seed-tagged duplicates so
    #    the report aligns with other classes)
    for i in range(n_per_type):
        cohort.append(AuditAdversarialProbe(
            name=f"unsealed_trail_v{i}",
            probe_type="chain",
            expected_attestor="chain",
            expected_kinds=("missing_hash",),
            mutator=_unsealed,
        ))

    # 7. runtime_drop at every blocked index
    for i in range(n_per_type):
        idx = base_indices[i % len(base_indices)]
        cohort.append(AuditAdversarialProbe(
            name=f"runtime_drop_at_{idx}_v{i}",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("missing_packet",),
            mutator=_make_runtime_drop_at(idx),
        ))

    # 8. runtime_extra (different phantom names)
    for i in range(n_per_type):
        cohort.append(AuditAdversarialProbe(
            name=f"runtime_extra_v{i}",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("extra_packet",),
            mutator=_make_runtime_extra(f"v{i}"),
        ))

    # 9. decision_flip at every index
    for i in range(n_per_type):
        idx = base_indices[i % len(base_indices)]
        cohort.append(AuditAdversarialProbe(
            name=f"decision_flip_at_{idx}_v{i}",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("decision_mismatch", "rules_mismatch"),
            mutator=_make_decision_flip_at(idx),
        ))

    # 10. rules_swap with different replacements
    replacements = [("wrong_rule",), ("rule_A", "rule_B"),
                    ("forged_block",), ("phantom_rule",),
                    ("override_block",), ("synthetic_rule",)]
    for i in range(n_per_type):
        idx = 2  # the blocked packet
        repl = replacements[i % len(replacements)]
        cohort.append(AuditAdversarialProbe(
            name=f"rules_swap_at_{idx}_{i}",
            probe_type="cross",
            expected_attestor="cross",
            expected_kinds=("rules_mismatch",),
            mutator=_make_rules_swap_at(idx, repl),
        ))

    return cohort


# -------------------------------------------------------------------- #
# Sweep                                                                #
# -------------------------------------------------------------------- #


def sweep_audit_adversarial(
    cohort: list[AuditAdversarialProbe],
) -> AuditAdversarialReport:
    """Execute every probe and report per-kind safety.

    A probe is "safe" when the named attestor reports at least one
    break whose ``kind`` field appears in ``expected_kinds`` ---
    extra (unrelated) breaks do not invalidate the verdict, but
    silence does.
    """
    per_n: dict[str, int] = {}
    per_safe: dict[str, int] = {}
    per_unsafe: dict[str, list[tuple[str, str]]] = {}

    for probe in cohort:
        per_n[probe.probe_type] = per_n.get(probe.probe_type, 0) + 1
        try:
            packets, rt = probe.mutator(*make_baseline_trail())
            chain_breaks = verify_audit_chain(packets)
            cross_breaks = cross_attest_policy(rt, packets)
        except Exception as exc:
            per_unsafe.setdefault(probe.probe_type, []).append(
                (probe.name, f"raised: {type(exc).__name__}: {exc}")
            )
            continue

        is_safe, evidence = _classify(probe, chain_breaks, cross_breaks)
        if is_safe:
            per_safe[probe.probe_type] = (
                per_safe.get(probe.probe_type, 0) + 1
            )
        else:
            per_unsafe.setdefault(probe.probe_type, []).append(
                (probe.name, evidence)
            )

    return AuditAdversarialReport(
        n_total=len(cohort),
        per_kind_n=per_n,
        per_kind_safe=per_safe,
        per_kind_unsafe=per_unsafe,
    )


def _classify(probe, chain_breaks, cross_breaks) -> tuple[bool, str]:
    if probe.expected_attestor == "chain":
        kinds = {b.kind for b in chain_breaks}
        if any(k in kinds for k in probe.expected_kinds):
            return True, f"chain reported {sorted(kinds)}"
        return False, (
            f"expected chain to report {list(probe.expected_kinds)}, "
            f"got chain={sorted(kinds)} cross={sorted({b.kind for b in cross_breaks})}"
        )

    if probe.expected_attestor == "cross":
        kinds = {b.kind for b in cross_breaks}
        if any(k in kinds for k in probe.expected_kinds):
            return True, f"cross reported {sorted(kinds)}"
        return False, (
            f"expected cross to report {list(probe.expected_kinds)}, "
            f"got cross={sorted(kinds)} chain={sorted({b.kind for b in chain_breaks})}"
        )

    # Both attestors must fire.
    chain_kinds = {b.kind for b in chain_breaks}
    cross_kinds = {b.kind for b in cross_breaks}
    if (any(k in chain_kinds for k in probe.expected_kinds)
            and cross_breaks):
        return True, f"chain={sorted(chain_kinds)} cross={sorted(cross_kinds)}"
    return False, (
        f"expected both, got chain={sorted(chain_kinds)} "
        f"cross={sorted(cross_kinds)}"
    )
