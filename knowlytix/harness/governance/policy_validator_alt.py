# SPDX-License-Identifier: Apache-2.0
"""Independent second-implementation validator.

A high-stakes deployment expects a *second* implementation of the
quorum logic --- written in a different style, ideally by a
different team --- that re-derives the BLOCK / ALLOW / REVIEW
verdict from the same audit record. The two implementations must
agree on every probe in the labeled cohort. Disagreement is a
defect that blocks ship.

This module is the in-repo realisation of that pattern. The
:class:`AltValidator` is *not* a wrapper around
``derive_attestor_verdict`` --- it implements the quorum logic
from scratch, using a boolean truth table over the four layer
flags. If the two implementations agree across the labeled cohort,
the auditor has structural evidence that the quorum logic is not
a single-implementation artefact.

For deployment-grade independence the second implementation must
be written in a different language (the chapter notes this). The
in-repo version is a stylistic alternative: list comprehensions vs
manual counters, explicit truth table vs derived counts, no shared
imports with :mod:`policy_calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Truth table: (l1, l2, l3, l4) -> {"BLOCK", "ALLOW", "REVIEW"}
#
# Each layer takes one of three values: True (block), False (pass),
# None (signal_unavailable). 36 reachable states (2*2*3*3).
# Rule: ALLOW iff zero blocks; BLOCK iff >= quorum; otherwise REVIEW.
# Signal-unavailable layers are excluded from the quorum count.
# ---------------------------------------------------------------------------


def _alt_quorum_decision(
    l1: bool, l2: bool,
    l3: bool | None, l4: bool | None,
    *, quorum: int = 2,
) -> str:
    """Truth-table-driven quorum verdict.

    Implementation note: this function deliberately does not call
    :func:`derive_attestor_verdict`; it is a from-scratch
    re-derivation so disagreement reveals a real logic divergence.
    """
    blocks = [l1, l2]
    if l3 is not None:
        blocks.append(l3)
    if l4 is not None:
        blocks.append(l4)
    n_block = sum(1 for b in blocks if b)
    if n_block == 0:
        return "ALLOW"
    if n_block >= quorum:
        return "BLOCK"
    return "REVIEW"


def _alt_extract_layers(
    blocking_rules: list[str],
    gms_scores: dict[str, float],
    *,
    tau_contra: float = 1.7,
) -> tuple[bool, bool, bool | None, bool | None]:
    """Re-derive layer flags from raw decision fields.

    Independent reading of the rule-id prefix conventions:
      * Layer 1: any rule whose id is neither ``gms_graph:*`` nor
        ``gms_plausibility:*``.
      * Layer 2: any ``gms_graph:*`` rule id.
      * Layer 3: any ``gms_plausibility:*`` rule id; signal
        available iff a ``plausibility`` score is in ``gms_scores``.
      * Layer 4: any ``tension:*`` score in ``gms_scores`` exceeds
        ``tau_contra``; signal available iff any tension score
        exists.
    """
    l1 = False
    l2 = False
    l3_block = False
    l3_avail = False
    for rid in blocking_rules:
        if rid.startswith("gms_graph:"):
            l2 = True
        elif rid.startswith("gms_plausibility:"):
            l3_block = True
            l3_avail = True
        else:
            l1 = True

    if "plausibility" in gms_scores:
        l3_avail = True

    tension_scores = [
        v for k, v in gms_scores.items()
        if k.startswith("tension:")
    ]
    if not tension_scores:
        l4: bool | None = None
    else:
        l4 = any(float(v) > tau_contra for v in tension_scores)

    l3: bool | None = l3_block if l3_avail else None
    return l1, l2, l3, l4


@dataclass
class AlternativeVerdict:
    decision: str
    layer1: bool
    layer2: bool
    layer3: bool | None
    layer4: bool | None
    n_block: int


def alt_check(
    blocking_rules: list[str],
    gms_scores: dict[str, float],
    *,
    quorum: int = 2,
    tau_contra: float = 1.7,
) -> AlternativeVerdict:
    """Stand-alone quorum check from raw decision fields.

    Used by :func:`alt_check_from_audit_record` for replay-grade
    re-derivation.
    """
    l1, l2, l3, l4 = _alt_extract_layers(
        blocking_rules, gms_scores, tau_contra=tau_contra,
    )
    decision = _alt_quorum_decision(l1, l2, l3, l4, quorum=quorum)
    n_block = (
        int(l1) + int(l2)
        + int(l3 is True) + int(l4 is True)
    )
    return AlternativeVerdict(
        decision=decision,
        layer1=l1, layer2=l2, layer3=l3, layer4=l4,
        n_block=n_block,
    )


def alt_check_from_audit_record(record) -> AlternativeVerdict:
    """Replay verdict from an archived audit record using the alt path."""
    return alt_check(
        list(record.blocking_rules),
        dict(record.gms_scores),
        quorum=record.quorum_required,
        tau_contra=1.7,
    )


# ---------------------------------------------------------------------------
# Cohort agreement check
# ---------------------------------------------------------------------------


@dataclass
class AgreementReport:
    n_total: int
    n_agree: int
    disagreements: list[tuple[str, str, str]]   # (probe_id, primary, alt)

    @property
    def agreement_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_agree / self.n_total

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        rows = [
            "AgreementReport(",
            f"  n_total={self.n_total}",
            f"  n_agree={self.n_agree}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for pid, primary, alt in self.disagreements[:5]:
            rows.append(
                f"    {pid}: primary={primary} alt={alt}"
            )
        rows.append(")")
        return "\n".join(rows)


def assert_engine_alt_agreement(
    engine,
    cohort,
    *,
    quorum_required: int = 2,
) -> AgreementReport:
    """Run engine + alt-validator on every probe; report agreement.

    A deployment ships only when this returns ``all_agree=True`` on
    the labeled cohort. Disagreement is a defect: the two
    implementations encode the quorum semantics differently somewhere.
    """
    from knowlytix.harness.governance.policy_calibration import check_with_quorum

    n_total = 0
    n_agree = 0
    disagreements: list[tuple[str, str, str]] = []

    for probe in cohort:
        decision, verdict = check_with_quorum(
            engine, probe.action, probe.args, probe.context,
            quorum_required=quorum_required,
        )
        alt = alt_check(
            list(decision.blocking_rules),
            dict(decision.gms_scores),
            quorum=quorum_required,
            tau_contra=float(getattr(
                engine, "TENSION_THRESHOLD",
                type(engine).TENSION_THRESHOLD,
            )),
        )
        n_total += 1
        if alt.decision == verdict.decision:
            n_agree += 1
        else:
            if len(disagreements) < 32:
                disagreements.append(
                    (probe.probe_id, verdict.decision, alt.decision)
                )

    return AgreementReport(
        n_total=n_total,
        n_agree=n_agree,
        disagreements=disagreements,
    )
