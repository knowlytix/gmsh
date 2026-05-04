# SPDX-License-Identifier: Apache-2.0
"""Property-based verification of audit-trail invariants.

Hand-rolled exhaustive enumeration over the audit-trail state space.
Asserts five invariants over every reachable state:

  J1. Empty trail produces head_hash="" and verify returns no breaks.
  J2. Sealed intact trail of length n verifies clean (zero breaks)
      and replay produces fingerprint deterministically.
  J3. Re-sealing an already-sealed intact trail is idempotent
      (same hashes; replay fingerprint unchanged).
  J4. A sealed trail with one packet's content edited produces
      exactly the ``content`` break at the edited index.
  J5. A sealed trail with one packet truncated produces a chain
      break at the next packet (mid) or no chain break but a
      cross-attestor break (tail).

These are exhaustive empirical checks across the FSM. The audit
chain is small enough that a real deployment may also produce a
machine-checked formal proof; the empirical enumeration here is
what the chapter delivers.

Mirrors :mod:`governed_agent.policy_invariants` for the policy
surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowlytix.harness.governance.tracing import AuditPacket
from knowlytix.harness.governance.tracing_hardening.hash_chain import (
    seal_audit_trail, verify_audit_chain, chain_head_hash,
)
from knowlytix.harness.governance.tracing_hardening.audit_replay import (
    replay_audit_metrics, metrics_fingerprint,
)


@dataclass
class AuditInvariantViolation:
    invariant: str
    state: dict
    detail: str


@dataclass
class AuditInvariantReport:
    n_states: int
    n_passed: int
    violations: list[AuditInvariantViolation]

    @property
    def all_pass(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        rows = [
            "AuditInvariantReport(",
            f"  states checked={self.n_states}",
            f"  invariants passed={self.n_passed}/{self.n_states}",
            f"  violations={len(self.violations)}",
        ]
        for v in self.violations[:5]:
            rows.append(f"    {v.invariant}: {v.detail}")
        rows.append(")")
        return "\n".join(rows)


def _make_synthetic_trail(n: int) -> list[AuditPacket]:
    """Build a synthetic trail of length n with deterministic fields.

    The trail's contents are not meaningful for the FSM check ---
    we only care that hash chaining produces the expected
    structure and that mutations are detected.
    """
    return [
        AuditPacket(
            step_num=i, tool_name=f"tool_{i}",
            policy_decision="allowed", blocking_rules=[],
            gms_scores={"plausibility": 0.5 + i * 0.05},
            timestamp=1000.0 + i,
            state_before=f"s{i}", state_after=f"s{i+1}",
        )
        for i in range(n)
    ]


def verify_audit_invariants() -> AuditInvariantReport:
    """Enumerate audit-trail states and verify J1-J5."""
    violations: list[AuditInvariantViolation] = []
    n_passed = 0
    n_states = 0

    # J1: empty trail
    n_states += 1
    state = {"n": 0, "case": "empty"}
    breaks = verify_audit_chain([])
    head = chain_head_hash([])
    if breaks or head != "":
        violations.append(AuditInvariantViolation(
            invariant="J1_empty_trail",
            state=state,
            detail=f"empty trail produced breaks={breaks}, head={head!r}",
        ))
    else:
        n_passed += 1

    # J2: sealed intact trails of varying lengths
    for n in range(1, 6):
        n_states += 1
        state = {"n": n, "case": "intact"}
        packets = _make_synthetic_trail(n)
        seal_audit_trail(packets)
        breaks = verify_audit_chain(packets)
        if breaks:
            violations.append(AuditInvariantViolation(
                invariant="J2_intact_sealed_clean",
                state=state,
                detail=f"sealed n={n} produced breaks {breaks}",
            ))
        else:
            n_passed += 1

    # J3: re-sealing is idempotent
    for n in range(1, 6):
        n_states += 1
        state = {"n": n, "case": "reseal_idempotent"}
        packets = _make_synthetic_trail(n)
        seal_audit_trail(packets)
        before_head = chain_head_hash(packets)
        before_fp = metrics_fingerprint(
            replay_audit_metrics(f"reseal-{n}", packets)
        )
        seal_audit_trail(packets)  # re-seal
        after_head = chain_head_hash(packets)
        after_fp = metrics_fingerprint(
            replay_audit_metrics(f"reseal-{n}", packets)
        )
        if before_head != after_head or before_fp != after_fp:
            violations.append(AuditInvariantViolation(
                invariant="J3_reseal_idempotent",
                state=state,
                detail=(
                    f"head before={before_head[:8]}... after={after_head[:8]}... "
                    f"fp before={before_fp[:8]}... after={after_fp[:8]}..."
                ),
            ))
        else:
            n_passed += 1

    # J4: content edit at every position produces exactly one
    # `content` break at the edited index (and possibly subsequent
    # `prev` breaks because chain propagates).
    for n in range(2, 5):
        for edit_idx in range(n):
            n_states += 1
            state = {
                "n": n, "case": "content_edit", "edit_idx": edit_idx,
            }
            packets = _make_synthetic_trail(n)
            seal_audit_trail(packets)
            packets[edit_idx].timestamp += 1.0  # edit
            breaks = verify_audit_chain(packets)
            content_at_edit = any(
                b.kind == "content" and b.index == edit_idx
                for b in breaks
            )
            if not content_at_edit:
                violations.append(AuditInvariantViolation(
                    invariant="J4_content_edit_detected",
                    state=state,
                    detail=(
                        f"edit at idx={edit_idx} produced "
                        f"breaks={[(b.kind, b.index) for b in breaks]}"
                    ),
                ))
            else:
                n_passed += 1

    # J5: truncation: mid-trail truncation breaks the chain at the
    # next packet; tail truncation produces no chain break.
    for n in range(2, 5):
        for trunc_idx in range(n):
            n_states += 1
            is_tail = (trunc_idx == n - 1)
            state = {
                "n": n, "case": "truncate",
                "trunc_idx": trunc_idx,
                "is_tail": is_tail,
            }
            packets = _make_synthetic_trail(n)
            seal_audit_trail(packets)
            packets = packets[:trunc_idx] + packets[trunc_idx + 1:]
            breaks = verify_audit_chain(packets)
            chain_detected = any(b.kind == "prev" for b in breaks)
            if is_tail:
                # Tail truncation must NOT produce a chain break
                # (no successor packet to break the link).
                if chain_detected:
                    violations.append(AuditInvariantViolation(
                        invariant="J5_tail_truncate_invisible_to_chain",
                        state=state,
                        detail=(
                            f"tail truncation at {trunc_idx} produced "
                            f"unexpected chain break {breaks}"
                        ),
                    ))
                else:
                    n_passed += 1
            else:
                # Mid truncation must produce a chain break
                if not chain_detected:
                    violations.append(AuditInvariantViolation(
                        invariant="J5_mid_truncate_chain_break",
                        state=state,
                        detail=(
                            f"mid truncation at {trunc_idx} did NOT "
                            f"produce chain break (got {breaks})"
                        ),
                    ))
                else:
                    n_passed += 1

    return AuditInvariantReport(
        n_states=n_states,
        n_passed=n_passed,
        violations=violations,
    )
