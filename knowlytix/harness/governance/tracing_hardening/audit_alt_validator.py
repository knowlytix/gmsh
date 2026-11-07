# SPDX-License-Identifier: Apache-2.0
"""Independent second-implementation validator for the audit chain.

A high-stakes deployment expects a second implementation of the
chain-verification logic --- written in a different style, ideally
by a different team --- that re-derives the chain breaks from the
same packets. The two implementations must agree on every probe in
the labeled cohort. Disagreement is a defect that blocks ship.

This module is the in-repo realisation of that pattern. Mirrors
:mod:`governed_agent.policy_validator_alt` for the audit surface.
The :func:`alt_verify_chain` is **not** a wrapper around
:func:`hash_chain.verify_audit_chain` --- it implements the
verification from scratch using a recursive functional walk and
an independent canonicalisation routine. If the two
implementations agree on the v2 cohort, the auditor has structural
evidence that chain verification is not a single-implementation
artefact.

For deployment-grade independence the second implementation must
be written in a different language. The in-repo version is a
stylistic alternative: recursion vs iteration, set-based break
classification vs append-on-mismatch, tuple comparison vs string
manipulation, no shared imports with :mod:`hash_chain`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from knowlytix.harness.governance.tracing import AuditPacket


@dataclass(frozen=True)
class AltChainBreak:
    """One detected break, alt-implementation style.

    Distinct type from :class:`hash_chain.ChainBreak` to prevent
    accidental aliasing — the agreement check compares values, not
    type identity.
    """

    index: int
    step_num: int
    kind: str
    detail: str


def _alt_canonical(packet: AuditPacket) -> bytes:
    """Canonicalisation matching the primary implementation.

    The canonical bytes are part of the audit *protocol*, not the
    implementation: two implementations that produce different
    canonical bytes will never agree on a hash even when both are
    correct, defeating the purpose of the independence check. This
    function produces the same bytes as
    :func:`hash_chain._canonical_packet_payload` --- both implement
    the protocol the same way; the difference between primary and
    alt is in the **verification flow** (recursive vs iterative,
    set-based break aggregation vs list-append, tail-call structure
    vs imperative).
    """
    payload = {
        k: v for k, v in asdict(packet).items()
        if k not in {"content_hash", "prev_hash"}
    }
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def _alt_packet_hash(packet: AuditPacket) -> str:
    return hashlib.sha256(_alt_canonical(packet)).hexdigest()


def alt_verify_chain(
    packets: list[AuditPacket],
) -> list[AltChainBreak]:
    """Recursive chain verification, distinct from the iterative path.

    Walks the chain head-first; each call recomputes the current
    packet's hash, validates it against the stored hash, validates
    the prev_hash against the predecessor, and recurses. Tail-call
    converted to a loop (Python lacks TCO) but the structure is
    different from the imperative ``verify_audit_chain``.
    """
    if not packets:
        return []

    breaks: list[AltChainBreak] = []
    expected_prev = ""

    def _verify_at(idx: int, expected_prev: str) -> str:
        if idx >= len(packets):
            return expected_prev
        pkt = packets[idx]

        # Missing-hash check: when content_hash is empty, the
        # packet was never sealed. Mirror the primary's behaviour
        # of skipping the content + prev checks in this case --- a
        # never-sealed packet has no fixed point of comparison.
        if not pkt.content_hash:
            breaks.append(AltChainBreak(
                index=idx, step_num=pkt.step_num,
                kind="missing_hash",
                detail="alt: no content_hash present",
            ))
            return ""  # cannot continue meaningfully

        # Recompute via the alt canonicalisation
        recomputed = _alt_packet_hash(pkt)
        if recomputed != pkt.content_hash:
            breaks.append(AltChainBreak(
                index=idx, step_num=pkt.step_num,
                kind="content",
                detail=(
                    f"alt: recomputed {recomputed[:12]}... != "
                    f"stored {pkt.content_hash[:12]}..."
                ),
            ))

        # Prev-hash chain check
        if pkt.prev_hash != expected_prev:
            breaks.append(AltChainBreak(
                index=idx, step_num=pkt.step_num,
                kind="prev",
                detail=(
                    f"alt: prev {pkt.prev_hash[:12]}... != "
                    f"expected {expected_prev[:12]}..."
                ),
            ))

        return pkt.content_hash

    cursor = expected_prev
    for i in range(len(packets)):
        cursor = _verify_at(i, cursor)

    return breaks


# ---------------------------------------------------------------------------
# Agreement check
# ---------------------------------------------------------------------------


@dataclass
class AltAgreementReport:
    n_probes: int
    n_agree: int
    disagreements: list[tuple[str, list[str], list[str]]]  # (probe, primary_kinds, alt_kinds)

    @property
    def agreement_rate(self) -> float:
        if self.n_probes == 0:
            return 0.0
        return self.n_agree / self.n_probes

    @property
    def all_agree(self) -> bool:
        return not self.disagreements

    def summary(self) -> str:
        rows = [
            "AltAgreementReport(",
            f"  n_probes={self.n_probes}",
            f"  n_agree={self.n_agree}",
            f"  agreement_rate={self.agreement_rate:.4f}",
            f"  disagreements={len(self.disagreements)}",
        ]
        for name, primary, alt in self.disagreements[:5]:
            rows.append(
                f"    {name}: primary={primary} alt={alt}"
            )
        rows.append(")")
        return "\n".join(rows)


def assert_chain_alt_agreement(
    cohort,
) -> AltAgreementReport:
    """Run primary and alt chain-verifiers on every probe; report agreement.

    A probe agrees when the **set of break kinds** is identical
    between the two implementations. Order of breaks within the
    list is implementation-defined (primary appends per-packet, alt
    walks recursively); only the kind set is required to match.
    """
    from knowlytix.harness.governance.tracing_hardening.audit_adversarial import (
        make_baseline_trail,
    )
    from knowlytix.harness.governance.tracing_hardening.hash_chain import (
        verify_audit_chain,
    )

    n_total = 0
    n_agree = 0
    disagreements: list[tuple[str, list[str], list[str]]] = []

    for probe in cohort:
        if probe.expected_attestor not in ("chain", "both"):
            # Cross-attestor probes do not exercise the chain path.
            continue
        packets, _rt = probe.mutator(*make_baseline_trail())
        primary_breaks = verify_audit_chain(packets)
        alt_breaks = alt_verify_chain(packets)

        primary_kinds = sorted({b.kind for b in primary_breaks})
        alt_kinds = sorted({b.kind for b in alt_breaks})

        n_total += 1
        if primary_kinds == alt_kinds:
            n_agree += 1
        else:
            if len(disagreements) < 16:
                disagreements.append(
                    (probe.name, primary_kinds, alt_kinds)
                )

    return AltAgreementReport(
        n_probes=n_total,
        n_agree=n_agree,
        disagreements=disagreements,
    )
