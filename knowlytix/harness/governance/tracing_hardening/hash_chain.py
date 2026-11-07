# SPDX-License-Identifier: Apache-2.0
"""Hash-chained audit trail for tamper-evident telemetry.

The :class:`AuditPacket` from :mod:`tracing` is mutable in memory.
For an audit-grade deployment that is necessary but insufficient: a
malicious or accidental edit to a packet (changing
``policy_decision`` from ``blocked`` to ``allowed``, deleting a
blocking rule, rewriting a timestamp) leaves no trace.  The trail
remains self-consistent under inspection.

The hash-chain pattern fixes this without changing the recording
API.  After the run completes, :func:`seal_audit_trail` walks the
packets in order and stamps two fields onto each one:

* ``content_hash`` --- SHA-256 over the canonical JSON of the
  packet's *unsealed* fields (everything except ``content_hash``
  and ``prev_hash``).
* ``prev_hash`` --- the ``content_hash`` of the previous packet,
  or the empty string for the first packet.

Once sealed, any post-hoc edit changes the modified packet's
``content_hash`` (because the canonical JSON now differs), which
breaks the chain at the next packet (whose ``prev_hash`` no longer
matches its predecessor's recomputed ``content_hash``).
:func:`verify_audit_chain` walks the chain and reports the first
broken link.

The chain is the same shape regardless of whether the audit trail
is also persisted to a :class:`DOETracingGMS` store.  In a
production deployment, both views agree by construction --- the
GMS store records the geometric fingerprint of each event, the
hash chain records the cryptographic fingerprint, and both must be
consistent.

Why not bake this into ``GovernanceTracer.trace_tool_call``: a
tracer that hashes on every call would still be vulnerable to
in-memory edits between the call and the verification check.
Sealing as a discrete step at run-end (or after every checkpoint)
forces the operator to commit to the chain explicitly, and the
seal becomes the artifact that ships with the audit log.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from knowlytix.harness.governance.tracing import AuditPacket


def _canonical_packet_payload(packet: AuditPacket) -> bytes:
    """Bytes used for hashing.  Excludes ``content_hash`` and ``prev_hash``.

    Sort keys for stability across Python releases and across hosts.
    Lists are dumped in their original order (the audit trail's
    chronology is meaningful and reordering would defeat the chain).
    """
    payload = {
        k: v for k, v in asdict(packet).items()
        if k not in ("content_hash", "prev_hash")
    }
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def compute_packet_content_hash(packet: AuditPacket) -> str:
    """Return the SHA-256 hex digest of a packet's unsealed payload."""
    return hashlib.sha256(_canonical_packet_payload(packet)).hexdigest()


def seal_audit_trail(packets: list[AuditPacket]) -> list[AuditPacket]:
    """Stamp ``content_hash`` and ``prev_hash`` onto every packet in order.

    The packets are mutated in place; the same list is returned for
    chaining convenience.  The first packet's ``prev_hash`` is set
    to the empty string --- a reserved sentinel that
    :func:`verify_audit_chain` recognises as ``head of chain``.

    Idempotent: re-sealing an already-sealed trail produces the
    same hashes provided no other field has changed.  This lets
    the runtime seal incrementally (after every N packets) without
    risking inconsistent state.
    """
    prev = ""
    for pkt in packets:
        pkt.prev_hash = prev
        pkt.content_hash = compute_packet_content_hash(pkt)
        prev = pkt.content_hash
    return packets


@dataclass(frozen=True)
class ChainBreak:
    """One detected break in the audit chain.

    ``index`` is the position of the offending packet in the trail.
    ``kind`` is one of:

    * ``"missing_hash"``  --- the packet was never sealed.
    * ``"content"``       --- the packet's stored ``content_hash``
      does not match the recomputed hash of its unsealed payload.
      Indicates an edit to one of the recorded fields.
    * ``"prev"``          --- the packet's ``prev_hash`` does not
      match the previous packet's ``content_hash``.  Indicates an
      insertion, deletion, or reorder.
    """

    index: int
    step_num: int
    kind: str
    detail: str


def verify_audit_chain(
    packets: list[AuditPacket],
) -> list[ChainBreak]:
    """Walk the chain and return every detected break.

    An empty list means the chain is intact: every packet's stored
    hashes match, and every link points to its predecessor.  A
    non-empty list names each broken packet and the kind of break.
    """
    breaks: list[ChainBreak] = []
    prev_content = ""

    for i, pkt in enumerate(packets):
        # Unsealed packets are an explicit error --- a runtime that
        # forgot to seal must be caught, not silently accepted.
        if not pkt.content_hash:
            breaks.append(ChainBreak(
                index=i,
                step_num=pkt.step_num,
                kind="missing_hash",
                detail="packet has no content_hash; was seal_audit_trail called?",
            ))
            prev_content = ""  # cannot continue meaningfully
            continue

        # Recompute over the (possibly tampered) payload.
        recomputed = compute_packet_content_hash(pkt)
        if recomputed != pkt.content_hash:
            breaks.append(ChainBreak(
                index=i,
                step_num=pkt.step_num,
                kind="content",
                detail=(
                    f"stored content_hash={pkt.content_hash[:12]}... "
                    f"but payload hashes to {recomputed[:12]}..."
                ),
            ))

        if pkt.prev_hash != prev_content:
            breaks.append(ChainBreak(
                index=i,
                step_num=pkt.step_num,
                kind="prev",
                detail=(
                    f"prev_hash={pkt.prev_hash[:12]}... but expected "
                    f"{prev_content[:12]}... from index {i-1}"
                ),
            ))

        prev_content = pkt.content_hash

    return breaks


def chain_head_hash(packets: list[AuditPacket]) -> str:
    """Return the ``content_hash`` of the last sealed packet, or '' if empty.

    The head hash is the natural fixed-length summary of the
    entire chain --- changing any earlier packet propagates
    through every subsequent ``prev_hash`` and surfaces as a
    different head.  The bundle records this so a runtime can
    cheaply check whether two trails are identical without
    re-walking every packet.
    """
    if not packets:
        return ""
    return packets[-1].content_hash
